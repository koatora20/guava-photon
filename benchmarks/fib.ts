// Fibonacci benchmark — TypeScript subset for guava-photon AOT
// This file can be compiled by guava-photon and also runs on Node.js

function fib(n: number): number {
  if (n <= 1) {
    return n;
  }
  return fib(n - 1) + fib(n - 2);
}
