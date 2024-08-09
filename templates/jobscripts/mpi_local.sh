/usr/bin/mpirun --bind-to none -np {{ num_procs }} {{ executable }} {{ input_file }} {{ output_file }}
