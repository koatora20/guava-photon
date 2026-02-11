// guava-photon AOT benchmark — Fibonacci via WebAssembly
// Run: node benchmarks/fib-wasm.mjs
// Requires: first compile with `node src/compiler.mjs benchmarks/fib.ts`

import { readFileSync } from 'fs';
import { execSync } from 'child_process';

// Compile TS → WAT → WASM
console.log('Compiling fib.ts → WebAssembly...');

// Step 1: TS → WAT (guava-photon AOT)
execSync('node src/compiler.mjs benchmarks/fib.ts', { stdio: 'inherit' });

// Step 2: WAT → WASM (requires wat2wasm from WABT)
try {
  execSync('wat2wasm benchmarks/fib.wat -o benchmarks/fib.wasm', { stdio: 'inherit' });
} catch {
  console.log('\n⚠️  wat2wasm not found. Install WABT: brew install wabt');
  console.log('   Or download from: https://github.com/WebAssembly/wabt/releases');
  console.log('\n   Showing WAT output instead:\n');
  console.log(readFileSync('benchmarks/fib.wat', 'utf-8'));
  process.exit(0);
}

// Step 3: Run WASM benchmark
const wasmBuffer = readFileSync('benchmarks/fib.wasm');
const { instance } = await WebAssembly.instantiate(wasmBuffer);

const N = 40;
const runs = 5;
const times = [];

for (let i = 0; i < runs; i++) {
  const start = performance.now();
  const result = instance.exports.fib(N);
  const elapsed = performance.now() - start;
  times.push(elapsed);
  if (i === 0) console.log(`fib(${N}) = ${result}`);
}

const avg = times.reduce((a, b) => a + b) / times.length;
const min = Math.min(...times);
console.log(`\nguava-photon AOT (WebAssembly):`);
console.log(`  Avg: ${avg.toFixed(2)}ms`);
console.log(`  Min: ${min.toFixed(2)}ms`);
console.log(`  Runs: ${runs}`);

// Compare with Node.js
console.log('\n--- Comparison ---');
console.log('Run both benchmarks:');
console.log('  node benchmarks/fib-node.mjs    # Node.js JIT');
console.log('  node benchmarks/fib-wasm.mjs    # guava-photon AOT');
