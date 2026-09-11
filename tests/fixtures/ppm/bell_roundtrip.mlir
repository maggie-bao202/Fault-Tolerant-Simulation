  func.func @qnode() {
    quantum.device shots(%c0_i64) ["/home/maggie_bao/Infleqtion/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst_0)
    %2 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %3 = pbc.ppr ["X"](4) %2 : !quantum.bit
    %4 = pbc.ppr ["Z"](4) %3 : !quantum.bit
    %5 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst)
    %6:2 = pbc.ppr ["Z", "X"](4) %4, %5 : !quantum.bit, !quantum.bit
    %7 = pbc.ppr ["Z"](-4) %6#0 : !quantum.bit
    %8 = pbc.ppr ["X"](-4) %6#1 : !quantum.bit
    quantum.gphase(%cst)
    %9:2 = pbc.ppr ["Z", "X"](4) %7, %8 : !quantum.bit, !quantum.bit
    %10 = pbc.ppr ["Z"](-4) %9#0 : !quantum.bit
    %11 = pbc.ppr ["X"](-4) %9#1 : !quantum.bit
    quantum.gphase(%cst_0)
    %12 = pbc.ppr ["Z"](4) %10 : !quantum.bit
    %13 = pbc.ppr ["X"](4) %12 : !quantum.bit
    %14 = pbc.ppr ["Z"](4) %13 : !quantum.bit
    %15 = quantum.namedobs %14[ PauliZ] : !quantum.obs
    %16 = quantum.expval %15 : f64
    %from_elements = tensor.from_elements %16 : tensor<f64>
    %17 = quantum.namedobs %11[ PauliZ] : !quantum.obs
    %18 = quantum.expval %17 : f64
    %from_elements_1 = tensor.from_elements %18 : tensor<f64>
    %19 = quantum.insert %0[ 0], %14 : !quantum.reg, !quantum.bit
    %20 = quantum.insert %19[ 1], %11 : !quantum.reg, !quantum.bit
    quantum.dealloc %20 : !quantum.reg
    quantum.device_release
  }
