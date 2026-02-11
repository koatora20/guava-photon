(module
  (func $fib (param $n f64) (result f64)
    (if (result f64) (i32.trunc_f64_s (select (f64.const 1) (f64.const 0) (f64.le (local.get $n) (f64.const 1))))
      (then
      (return (local.get $n))
        (f64.const 0)
      )
      (else (f64.const 0))
    )
    drop
    (return (f64.add (call $fib (f64.sub (local.get $n) (f64.const 1))) (call $fib (f64.sub (local.get $n) (f64.const 2)))))
    (f64.const 0)
  )
  (export "fib" (func $fib))
)
