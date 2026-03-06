;; guava-photon SIMD Tensor Engine v3 — 4-Way Unrolled f64x2
;; Processes 8 f64 values per loop iteration (4 × v128 = 4 × 2 × f64)
;; Theoretical: 4x over basic SIMD → ~8x vs Node.js at scale
;;
;; Memory: 256 pages = 16MB (1M-dim × 2 vectors)

(module
  (memory (export "memory") 256)

  ;; ============================================================
  ;; unrolledDotProduct: 4-way unrolled SIMD dot product
  ;; Processes 8 elements per iteration using 4 SIMD accumulators
  ;; Reduces loop overhead by 4x and maximizes ILP (instruction-level parallelism)
  ;; ============================================================
  (func $unrolledDotProduct (param $offsetA i32) (param $offsetB i32) (param $len i32) (result f64)
    (local $i i32)
    (local $sum0 v128)
    (local $sum1 v128)
    (local $sum2 v128)
    (local $sum3 v128)
    (local $end8 i32)
    (local $end2 i32)
    (local $scalarSum f64)
    (local $byteOffA i32)
    (local $byteOffB i32)
    
    ;; Initialize 4 independent accumulators (maximizes out-of-order execution)
    (local.set $sum0 (f64x2.splat (f64.const 0)))
    (local.set $sum1 (f64x2.splat (f64.const 0)))
    (local.set $sum2 (f64x2.splat (f64.const 0)))
    (local.set $sum3 (f64x2.splat (f64.const 0)))
    
    ;; Phase 1: Process 8 elements per iteration (4 × f64x2)
    (local.set $end8 (i32.and (local.get $len) (i32.const -8)))
    (local.set $i (i32.const 0))
    
    (block $break8
      (loop $unrolled_loop
        (br_if $break8 (i32.ge_u (local.get $i) (local.get $end8)))
        
        ;; Compute byte offsets
        (local.set $byteOffA (i32.add (local.get $offsetA) (i32.mul (local.get $i) (i32.const 8))))
        (local.set $byteOffB (i32.add (local.get $offsetB) (i32.mul (local.get $i) (i32.const 8))))
        
        ;; Acc 0: elements [i, i+1]
        (local.set $sum0
          (f64x2.add (local.get $sum0)
            (f64x2.mul
              (v128.load (local.get $byteOffA))
              (v128.load (local.get $byteOffB)))))
        
        ;; Acc 1: elements [i+2, i+3]
        (local.set $sum1
          (f64x2.add (local.get $sum1)
            (f64x2.mul
              (v128.load (i32.add (local.get $byteOffA) (i32.const 16)))
              (v128.load (i32.add (local.get $byteOffB) (i32.const 16))))))
        
        ;; Acc 2: elements [i+4, i+5]
        (local.set $sum2
          (f64x2.add (local.get $sum2)
            (f64x2.mul
              (v128.load (i32.add (local.get $byteOffA) (i32.const 32)))
              (v128.load (i32.add (local.get $byteOffB) (i32.const 32))))))
        
        ;; Acc 3: elements [i+6, i+7]
        (local.set $sum3
          (f64x2.add (local.get $sum3)
            (f64x2.mul
              (v128.load (i32.add (local.get $byteOffA) (i32.const 48)))
              (v128.load (i32.add (local.get $byteOffB) (i32.const 48))))))
        
        ;; i += 8
        (local.set $i (i32.add (local.get $i) (i32.const 8)))
        (br $unrolled_loop)
      )
    )
    
    ;; Merge 4 accumulators: sum0 + sum1 + sum2 + sum3
    (local.set $sum0 (f64x2.add (local.get $sum0) (local.get $sum1)))
    (local.set $sum2 (f64x2.add (local.get $sum2) (local.get $sum3)))
    (local.set $sum0 (f64x2.add (local.get $sum0) (local.get $sum2)))
    
    ;; Horizontal sum
    (local.set $scalarSum
      (f64.add
        (f64x2.extract_lane 0 (local.get $sum0))
        (f64x2.extract_lane 1 (local.get $sum0))))
    
    ;; Phase 2: Handle remaining elements (2 at a time with basic SIMD)
    (local.set $end2 (i32.and (local.get $len) (i32.const -2)))
    
    (block $break2
      (loop $simd_tail
        (br_if $break2 (i32.ge_u (local.get $i) (local.get $end2)))
        
        (local.set $scalarSum
          (f64.add (local.get $scalarSum)
            (f64.add
              (f64.mul
                (f64.load (i32.add (local.get $offsetA) (i32.mul (local.get $i) (i32.const 8))))
                (f64.load (i32.add (local.get $offsetB) (i32.mul (local.get $i) (i32.const 8)))))
              (f64.mul
                (f64.load (i32.add (local.get $offsetA) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 8))))
                (f64.load (i32.add (local.get $offsetB) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 8))))))))
        
        (local.set $i (i32.add (local.get $i) (i32.const 2)))
        (br $simd_tail)
      )
    )
    
    ;; Phase 3: Handle final odd element
    (if (i32.lt_u (local.get $i) (local.get $len))
      (then
        (local.set $scalarSum
          (f64.add (local.get $scalarSum)
            (f64.mul
              (f64.load (i32.add (local.get $offsetA) (i32.mul (local.get $i) (i32.const 8))))
              (f64.load (i32.add (local.get $offsetB) (i32.mul (local.get $i) (i32.const 8)))))))))
    
    (local.get $scalarSum)
  )
  
  ;; ============================================================
  ;; simdDotProduct: Basic SIMD (2 at a time) — reference
  ;; ============================================================
  (func $simdDotProduct (param $offsetA i32) (param $offsetB i32) (param $len i32) (result f64)
    (local $i i32)
    (local $sum v128)
    (local $end i32)
    (local $scalarSum f64)
    
    (local.set $sum (f64x2.splat (f64.const 0)))
    (local.set $end (i32.and (local.get $len) (i32.const -2)))
    (local.set $i (i32.const 0))
    
    (block $break
      (loop $simd_loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $end)))
        (local.set $sum
          (f64x2.add (local.get $sum)
            (f64x2.mul
              (v128.load (i32.add (local.get $offsetA) (i32.mul (local.get $i) (i32.const 8))))
              (v128.load (i32.add (local.get $offsetB) (i32.mul (local.get $i) (i32.const 8)))))))
        (local.set $i (i32.add (local.get $i) (i32.const 2)))
        (br $simd_loop)
      )
    )
    
    (local.set $scalarSum
      (f64.add
        (f64x2.extract_lane 0 (local.get $sum))
        (f64x2.extract_lane 1 (local.get $sum))))
    
    (if (i32.lt_u (local.get $end) (local.get $len))
      (then
        (local.set $scalarSum
          (f64.add (local.get $scalarSum)
            (f64.mul
              (f64.load (i32.add (local.get $offsetA) (i32.mul (local.get $end) (i32.const 8))))
              (f64.load (i32.add (local.get $offsetB) (i32.mul (local.get $end) (i32.const 8)))))))))
    
    (local.get $scalarSum)
  )
  
  ;; ============================================================
  ;; scalarDotProduct: Non-SIMD reference
  ;; ============================================================
  (func $scalarDotProduct (param $offsetA i32) (param $offsetB i32) (param $len i32) (result f64)
    (local $i i32)
    (local $sum f64)
    (local.set $sum (f64.const 0))
    (local.set $i (i32.const 0))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))
        (local.set $sum
          (f64.add (local.get $sum)
            (f64.mul
              (f64.load (i32.add (local.get $offsetA) (i32.mul (local.get $i) (i32.const 8))))
              (f64.load (i32.add (local.get $offsetB) (i32.mul (local.get $i) (i32.const 8)))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $sum)
  )
  
  ;; Exports
  (export "unrolledDotProduct" (func $unrolledDotProduct))
  (export "simdDotProduct" (func $simdDotProduct))
  (export "scalarDotProduct" (func $scalarDotProduct))
)
