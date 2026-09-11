  func.func @qnode() {
    quantum.device shots(%c0_i64) ["/home/maggie_bao/Infleqtion/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst_0)
    %2 = pbc.ppr ["Z"](8) %1 : !quantum.bit
    %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst)
    %4:2 = pbc.ppr ["Z", "X"](4) %2, %3 : !quantum.bit, !quantum.bit
    %5 = pbc.ppr ["Z"](-4) %4#0 : !quantum.bit
    %6 = pbc.ppr ["X"](-4) %4#1 : !quantum.bit
    %7 = quantum.insert %0[ 0], %5 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %6 : !quantum.reg, !quantum.bit
    %9 = catalyst.list_init : <f64>
    %10 = catalyst.list_init : <i64>
    catalyst.list_push %cst_0, %9 : <f64>
    %11 = quantum.extract %8[ 1] : !quantum.reg -> !quantum.bit
    %12 = pbc.ppr ["Z"](-8) %11 : !quantum.bit
    %13 = catalyst.list_pop %9 : <f64>
    quantum.gphase(%13) adj
    %14 = quantum.insert %8[ 1], %12 : !quantum.reg, !quantum.bit
    catalyst.list_dealloc %9 : <f64>
    catalyst.list_dealloc %10 : <i64>
    %15 = quantum.extract %14[ 0] : !quantum.reg -> !quantum.bit
    %16 = quantum.namedobs %15[ PauliZ] : !quantum.obs
    %17 = quantum.expval %16 : f64
    %from_elements = tensor.from_elements %17 : tensor<f64>
    %18 = quantum.extract %14[ 1] : !quantum.reg -> !quantum.bit
    %19 = quantum.namedobs %18[ PauliZ] : !quantum.obs
    %20 = quantum.expval %19 : f64
    %from_elements_1 = tensor.from_elements %20 : tensor<f64>
    %21 = quantum.insert %14[ 0], %15 : !quantum.reg, !quantum.bit
    %22 = quantum.insert %21[ 1], %18 : !quantum.reg, !quantum.bit
    quantum.dealloc %22 : !quantum.reg
    quantum.device_release
  }
