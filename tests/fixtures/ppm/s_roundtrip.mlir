  func.func @qnode() {
    quantum.device shots(%c0_i64) ["/home/maggie_bao/Infleqtion/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst_1)
    %2 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %3 = pbc.ppr ["X"](4) %2 : !quantum.bit
    %4 = pbc.ppr ["Z"](4) %3 : !quantum.bit
    %5 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst_0)
    %6:2 = pbc.ppr ["Z", "X"](4) %4, %5 : !quantum.bit, !quantum.bit
    %7 = pbc.ppr ["Z"](-4) %6#0 : !quantum.bit
    %8 = pbc.ppr ["X"](-4) %6#1 : !quantum.bit
    quantum.gphase(%cst)
    %9 = pbc.ppr ["Z"](4) %8 : !quantum.bit
    %10 = quantum.insert %0[ 0], %7 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[ 1], %9 : !quantum.reg, !quantum.bit
    %12 = catalyst.list_init : <f64>
    %13 = catalyst.list_init : <i64>
    catalyst.list_push %cst, %12 : <f64>
    %14 = quantum.extract %11[ 1] : !quantum.reg -> !quantum.bit
    %15 = pbc.ppr ["Z"](-4) %14 : !quantum.bit
    %16 = catalyst.list_pop %12 : <f64>
    quantum.gphase(%16) adj
    %17 = quantum.insert %11[ 1], %15 : !quantum.reg, !quantum.bit
    catalyst.list_dealloc %12 : <f64>
    catalyst.list_dealloc %13 : <i64>
    %18 = quantum.extract %17[ 0] : !quantum.reg -> !quantum.bit
    %19 = quantum.extract %17[ 1] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst_0)
    %20:2 = pbc.ppr ["Z", "X"](4) %18, %19 : !quantum.bit, !quantum.bit
    %21 = pbc.ppr ["Z"](-4) %20#0 : !quantum.bit
    %22 = pbc.ppr ["X"](-4) %20#1 : !quantum.bit
    quantum.gphase(%cst_1)
    %23 = pbc.ppr ["Z"](4) %21 : !quantum.bit
    %24 = pbc.ppr ["X"](4) %23 : !quantum.bit
    %25 = pbc.ppr ["Z"](4) %24 : !quantum.bit
    %26 = quantum.namedobs %25[ PauliZ] : !quantum.obs
    %27 = quantum.expval %26 : f64
    %from_elements = tensor.from_elements %27 : tensor<f64>
    %28 = quantum.namedobs %22[ PauliZ] : !quantum.obs
    %29 = quantum.expval %28 : f64
    %from_elements_2 = tensor.from_elements %29 : tensor<f64>
    %30 = quantum.insert %17[ 0], %25 : !quantum.reg, !quantum.bit
    %31 = quantum.insert %30[ 1], %22 : !quantum.reg, !quantum.bit
    quantum.dealloc %31 : !quantum.reg
    quantum.device_release
  }
