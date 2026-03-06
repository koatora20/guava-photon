// guava-photon Node.js API Compatibility Layer
// Minimal shim implementing the most critical Node.js APIs
// Strategy: passthrough to native Node.js for I/O, WASM for computation
//
// Phase 1: fs, path, process, Buffer (session-load minimum)
// Phase 2: http, net, crypto (full runtime)

import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'fs';
import { resolve, join, dirname, basename, extname, relative, sep } from 'path';

// ============================================================
// Photon Runtime: bridges Node.js APIs with WASM tensor engine
// ============================================================

export class PhotonRuntime {
    constructor(wasmPath) {
        this.wasmPath = wasmPath;
        this.instance = null;
        this.memory = null;
        this.startTime = performance.now();
    }

    async init() {
        if (this.wasmPath && existsSync(this.wasmPath)) {
            const buf = readFileSync(this.wasmPath);
            const result = await WebAssembly.instantiate(buf);
            this.instance = result.instance;
            this.memory = this.instance.exports.memory;
        }
        return this;
    }

    // ============================================================
    // node:fs shim (passthrough + WASM-accelerated operations)
    // ============================================================
    get fs() {
        const runtime = this;
        return {
            readFileSync(path, encoding) {
                return readFileSync(path, encoding);
            },
            writeFileSync(path, data) {
                writeFileSync(path, data);
            },
            existsSync(path) {
                return existsSync(path);
            },
            mkdirSync(path, options) {
                return mkdirSync(path, options);
            },
            readdirSync(path) {
                return readdirSync(path);
            },
            statSync(path) {
                return statSync(path);
            },
            // WASM-accelerated: read file into WASM memory
            readToWasm(path, memoryOffset) {
                const data = readFileSync(path);
                const view = new Uint8Array(runtime.memory.buffer, memoryOffset, data.length);
                view.set(data);
                return data.length;
            },
            // WASM-accelerated: read float64 array from file into WASM memory
            readFloat64ToWasm(path, memoryOffset) {
                const data = readFileSync(path);
                const f64 = new Float64Array(data.buffer, data.byteOffset, data.byteLength / 8);
                const view = new Float64Array(runtime.memory.buffer, memoryOffset, f64.length);
                view.set(f64);
                return f64.length;
            }
        };
    }

    // ============================================================
    // node:path shim (pure passthrough)
    // ============================================================
    get path() {
        return { resolve, join, dirname, basename, extname, relative, sep };
    }

    // ============================================================
    // process shim (minimal)
    // ============================================================
    get process() {
        const runtime = this;
        return {
            get argv() { return process.argv; },
            get env() { return process.env; },
            get cwd() { return () => process.cwd(); },
            get platform() { return process.platform; },
            get arch() { return process.arch; },
            exit(code) { process.exit(code); },
            // WASM-accelerated: high-res timing via WASM
            hrtime() {
                return performance.now() - runtime.startTime;
            }
        };
    }

    // ============================================================
    // Buffer shim (bridges to WASM memory)
    // ============================================================
    get Buffer() {
        const runtime = this;
        return {
            alloc(size) {
                return new Uint8Array(size);
            },
            from(data, encoding) {
                if (typeof data === 'string') {
                    return new TextEncoder().encode(data);
                }
                return new Uint8Array(data);
            },
            // Copy to WASM memory
            toWasm(buffer, offset) {
                const view = new Uint8Array(runtime.memory.buffer, offset, buffer.length);
                view.set(buffer);
                return buffer.length;
            },
            // Copy from WASM memory
            fromWasm(offset, length) {
                return new Uint8Array(runtime.memory.buffer, offset, length);
            }
        };
    }

    // ============================================================
    // Tensor API: direct access to WASM functions
    // ============================================================
    get tensor() {
        const inst = this.instance;
        if (!inst) throw new Error('WASM module not loaded');

        return {
            // Fill vector into WASM memory
            fill(values, memoryOffset) {
                const view = new Float64Array(inst.exports.memory.buffer, memoryOffset, values.length);
                view.set(values);
            },

            // Core tensor operations (WASM-accelerated)
            dotProduct(offsetA, offsetB, len) {
                return inst.exports.unrolledDotProduct
                    ? inst.exports.unrolledDotProduct(offsetA, offsetB, len)
                    : inst.exports.simdDotProduct
                        ? inst.exports.simdDotProduct(offsetA, offsetB, len)
                        : inst.exports.scalarDotProduct(offsetA, offsetB, len);
            },

            // High-level: compute similarity between two JS arrays
            similarity(a, b) {
                const N = a.length;
                const mem = new Float64Array(inst.exports.memory.buffer);
                mem.set(a, 0);
                mem.set(b, N);

                const dot = this.dotProduct(0, N * 8, N);
                const magA = Math.sqrt(this.dotProduct(0, 0, N));
                const magB = Math.sqrt(this.dotProduct(N * 8, N * 8, N));

                return magA * magB > 0 ? dot / (magA * magB) : 0;
            },

            // Available functions
            get exports() {
                return Object.keys(inst.exports).filter(k => typeof inst.exports[k] === 'function');
            }
        };
    }
}

// ============================================================
// Worker Pool: persistent WASM instances across workers
// ============================================================

import { Worker, isMainThread, workerData, parentPort } from 'worker_threads';
import { fileURLToPath } from 'url';

if (!isMainThread && workerData?.type === 'photon-pool-worker') {
    // Pool worker: stays alive and processes tasks
    const runtime = new PhotonRuntime(workerData.wasmPath);
    await runtime.init();

    parentPort.on('message', (msg) => {
        if (msg.type === 'dotProduct') {
            const { vectorA, vectorB, id } = msg;
            const N = vectorA.length;
            runtime.tensor.fill(vectorA, 0);
            runtime.tensor.fill(vectorB, N * 8);
            const result = runtime.tensor.dotProduct(0, N * 8, N);
            parentPort.postMessage({ id, result });
        }
        if (msg.type === 'shutdown') {
            process.exit(0);
        }
    });
}

export class PhotonWorkerPool {
    constructor(wasmPath, numWorkers = 4) {
        this.wasmPath = wasmPath;
        this.numWorkers = numWorkers;
        this.workers = [];
        this.pending = new Map();
        this.nextId = 0;
        this.roundRobin = 0;
    }

    async init() {
        const workerFile = fileURLToPath(import.meta.url);

        for (let i = 0; i < this.numWorkers; i++) {
            const worker = new Worker(workerFile, {
                workerData: { type: 'photon-pool-worker', wasmPath: this.wasmPath }
            });

            worker.on('message', (msg) => {
                const resolver = this.pending.get(msg.id);
                if (resolver) {
                    resolver(msg.result);
                    this.pending.delete(msg.id);
                }
            });

            this.workers.push(worker);
        }

        // Wait for workers to initialize
        await new Promise(r => setTimeout(r, 100));
        return this;
    }

    async dotProduct(vectorA, vectorB) {
        const id = this.nextId++;
        const workerIdx = this.roundRobin++ % this.numWorkers;

        return new Promise((resolve) => {
            this.pending.set(id, resolve);
            this.workers[workerIdx].postMessage({
                type: 'dotProduct', vectorA, vectorB, id
            });
        });
    }

    // Parallel dot product: split across all workers
    async parallelDotProduct(vectorA, vectorB) {
        const N = vectorA.length;
        const chunkSize = Math.ceil(N / this.numWorkers);
        const promises = [];

        for (let i = 0; i < this.numWorkers; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, N);
            if (start >= N) break;

            promises.push(this.dotProduct(
                vectorA.slice(start, end),
                vectorB.slice(start, end)
            ));
        }

        const partials = await Promise.all(promises);
        return partials.reduce((a, b) => a + b, 0);
    }

    async shutdown() {
        for (const worker of this.workers) {
            worker.postMessage({ type: 'shutdown' });
        }
        this.workers = [];
    }
}
