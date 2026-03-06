// guava-photon Runner: AOT Pipeline Orchestrator
// Compiles TypeScript → WAT → WASM and executes cognitive functions
//
// Usage:
//   node src/runner.mjs compile src/tensor.ts    # TS → WAT → WASM
//   node src/runner.mjs bench src/tensor.ts      # Compile + benchmark
//   node src/runner.mjs run src/tensor.ts <fn> [args...]  # Run function

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { resolve, basename } from 'path';

const ROOT = resolve(import.meta.dirname, '..');

// ============================================================
// Pipeline: TS → WAT → WASM
// ============================================================

function compile(inputPath) {
    const absInput = resolve(inputPath);
    const watPath = absInput.replace(/\.(ts|js|mjs)$/, '.wat');
    const wasmPath = absInput.replace(/\.(ts|js|mjs)$/, '.wasm');

    // Step 1: TS → WAT via guava-photon compiler
    const compilerPath = resolve(ROOT, 'src/compiler.mjs');
    console.log(`🔬 Step 1: ${basename(absInput)} → WAT`);
    execSync(`node ${compilerPath} ${absInput}`, { stdio: 'inherit' });

    // Step 2: WAT → WASM via wat2wasm
    console.log(`🔬 Step 2: WAT → WASM`);
    try {
        execSync(`wat2wasm ${watPath} -o ${wasmPath}`, { stdio: 'pipe' });
    } catch {
        // Try /tmp as fallback for permission issues
        const tmpWasm = `/tmp/${basename(wasmPath)}`;
        execSync(`wat2wasm ${watPath} -o ${tmpWasm}`, { stdio: 'pipe' });
        console.log(`   (saved to ${tmpWasm} due to permissions)`);
        return tmpWasm;
    }

    const watSize = readFileSync(watPath).length;
    const wasmSize = readFileSync(wasmPath).length;
    console.log(`✅ Pipeline complete: ${basename(absInput)}`);
    console.log(`   WAT: ${watSize} bytes → WASM: ${wasmSize} bytes`);
    return wasmPath;
}

// ============================================================
// Load WASM module
// ============================================================

async function loadModule(wasmPath) {
    const buffer = readFileSync(wasmPath);
    const { instance } = await WebAssembly.instantiate(buffer);
    return instance;
}

// ============================================================
// Main
// ============================================================

const [, , command, input, ...args] = process.argv;

if (!command) {
    console.log('guava-photon Runner v2');
    console.log('');
    console.log('Commands:');
    console.log('  compile <file.ts>              Compile TS → WAT → WASM');
    console.log('  run <file.ts> <func> [args]    Compile + run function');
    console.log('  simd                           Load SIMD tensor module');
    console.log('');
    process.exit(0);
}

if (command === 'compile') {
    if (!input) { console.error('Error: input file required'); process.exit(1); }
    compile(input);
}

if (command === 'run') {
    if (!input || !args[0]) { console.error('Usage: run <file.ts> <function> [args]'); process.exit(1); }
    const wasmPath = compile(input);
    const instance = await loadModule(wasmPath);
    const fn = args[0];
    const fnArgs = args.slice(1).map(Number);

    if (!instance.exports[fn]) {
        console.error(`Function "${fn}" not found. Available: ${Object.keys(instance.exports).filter(k => typeof instance.exports[k] === 'function').join(', ')}`);
        process.exit(1);
    }

    const result = instance.exports[fn](...fnArgs);
    console.log(`\n🍈 ${fn}(${fnArgs.join(', ')}) = ${result}`);
}

if (command === 'simd') {
    const simdWat = resolve(ROOT, 'src/simd-tensor.wat');
    const simdWasm = '/tmp/simd-tensor.wasm';

    console.log('🔬 Compiling SIMD tensor module...');
    execSync(`wat2wasm ${simdWat} -o ${simdWasm}`, { stdio: 'pipe' });

    const instance = await loadModule(simdWasm);
    const exports = Object.keys(instance.exports).filter(k => typeof instance.exports[k] === 'function');
    console.log(`✅ SIMD module loaded: ${exports.join(', ')}`);
    console.log(`   Memory: ${instance.exports.memory.buffer.byteLength / 1024 / 1024}MB`);
}
