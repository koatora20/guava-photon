// Node.js JIT benchmark — Fibonacci
// Run: node benchmarks/fib-node.mjs

function fib(n) {
  if (n <= 1) return n;
  return fib(n - 1) + fib(n - 2);
}

const N = 40;
const runs = 5;
const times = [];

for (let i = 0; i < runs; i++) {
  const start = performance.now();
  const result = fib(N);
  const elapsed = performance.now() - start;
  times.push(elapsed);
  if (i === 0) console.log(`fib(${N}) = ${result}`);
}

const avg = times.reduce((a, b) => a + b) / times.length;
const min = Math.min(...times);
console.log(`\nNode.js JIT (V8):`);
console.log(`  Avg: ${avg.toFixed(2)}ms`);
console.log(`  Min: ${min.toFixed(2)}ms`);
console.log(`  Runs: ${runs}`);
