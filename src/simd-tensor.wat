;; guava-photon SIMD Tensor Engine — Hand-written WAT
;; WASM SIMD 128-bit f64x2 operations for batched tensor computations
;; This module bypasses the compiler for maximum SIMD control
;;
;; Memory layout: 
;;   [0 .. 8*N)       = vector A (f64 array)
;;   [8*N .. 16*N)    = vector B (f64 array)
;;   Result returned as f64 scalar

(module
  ;; 1 page = 64KB = 8192 f64 values (enough for 4096-dim vectors × 2)
  (memory (export "memory") 16)

  ;; ============================================================
  ;; simdDotProduct: SIMD-accelerated dot product
  ;; Processes 2 f64 values per iteration using f64x2
  ;; Args: offsetA (i32), offsetB (i32), length (i32)
  ;; Returns: f64 (dot product)
  ;; ============================================================
  (func $simdDotProduct (param $offsetA i32) (param $offsetB i32) (param $len i32) (result f64)
    (local $i i32)
    (local $sum v128)
    (local $end i32)
    (local $scalarSum f64)
    
    ;; Initialize sum vector to zero
    (local.set $sum (f64x2.splat (f64.const 0)))
    
    ;; Process pairs of f64 using SIMD (16 bytes = 2 × f64 per iteration)
    ;; end = len & ~1 (round down to even)
    (local.set $end (i32.and (local.get $len) (i32.const -2)))
    (local.set $i (i32.const 0))
    
    (block $break
      (loop $simd_loop
        ;; Break if i >= end
        (br_if $break (i32.ge_u (local.get $i) (local.get $end)))
        
        ;; Load 2 f64 values from A: v128.load(offsetA + i * 8)
        ;; Load 2 f64 values from B: v128.load(offsetB + i * 8)
        ;; Multiply: f64x2.mul(a_pair, b_pair)
        ;; Accumulate: f64x2.add(sum, product)
        (local.set $sum
          (f64x2.add
            (local.get $sum)
            (f64x2.mul
              (v128.load (i32.add (local.get $offsetA) (i32.mul (local.get $i) (i32.const 8))))
              (v128.load (i32.add (local.get $offsetB) (i32.mul (local.get $i) (i32.const 8))))
            )
          )
        )
        
        ;; i += 2
        (local.set $i (i32.add (local.get $i) (i32.const 2)))
        (br $simd_loop)
      )
    )
    
    ;; Horizontal sum: extract both lanes and add
    (local.set $scalarSum
      (f64.add
        (f64x2.extract_lane 0 (local.get $sum))
        (f64x2.extract_lane 1 (local.get $sum))
      )
    )
    
    ;; Handle odd remainder (if len is odd)
    (if (i32.lt_u (local.get $end) (local.get $len))
      (then
        (local.set $scalarSum
          (f64.add
            (local.get $scalarSum)
            (f64.mul
              (f64.load (i32.add (local.get $offsetA) (i32.mul (local.get $end) (i32.const 8))))
              (f64.load (i32.add (local.get $offsetB) (i32.mul (local.get $end) (i32.const 8))))
            )
          )
        )
      )
    )
    
    (local.get $scalarSum)
  )
  
  ;; ============================================================
  ;; scalarDotProduct: Non-SIMD reference for comparison
  ;; Args: offsetA (i32), offsetB (i32), length (i32)
  ;; Returns: f64 (dot product)
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
          (f64.add
            (local.get $sum)
            (f64.mul
              (f64.load (i32.add (local.get $offsetA) (i32.mul (local.get $i) (i32.const 8))))
              (f64.load (i32.add (local.get $offsetB) (i32.mul (local.get $i) (i32.const 8))))
            )
          )
        )
        
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    
    (local.get $sum)
  )
  
  ;; ============================================================
  ;; simdMagnitude: SIMD-accelerated L2 norm (sqrt of sum of squares)
  ;; Args: offset (i32), length (i32)
  ;; Returns: f64 (magnitude)
  ;; ============================================================
  (func $simdMagnitude (param $offset i32) (param $len i32) (result f64)
    ;; Reuse dot product: magnitude = sqrt(dot(a, a))
    (f64.sqrt (call $simdDotProduct (local.get $offset) (local.get $offset) (local.get $len)))
  )
  
  ;; Export all functions
  (export "simdDotProduct" (func $simdDotProduct))
  (export "scalarDotProduct" (func $scalarDotProduct))
  (export "simdMagnitude" (func $simdMagnitude))
)
