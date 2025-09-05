sudo -E "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" \
    /usr/local/cuda/bin/ncu \
    -o report_backward \
    --target-processes all \
    --set full \
    -k pos_mlp_bias_backward_kernel \
    -f python3 test.py

