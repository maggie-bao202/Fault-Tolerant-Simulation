  func.func @qnode() {
    quantum.device shots(%c0_i64) ["/home/maggie_bao/Infleqtion/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    quantum.gphase(%cst)
    %2 = pbc.ppr ["Z"](8) %1 : !quantum.bit
    %3 = quantum.namedobs %2[ PauliZ] : !quantum.obs
    %4 = quantum.expval %3 : f64
    %from_elements = tensor.from_elements %4 : tensor<f64>
    %5 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
    quantum.device_release
  }
