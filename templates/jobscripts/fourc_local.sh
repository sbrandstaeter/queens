{{ mpi_cmd }} -np {{ num_procs }} {{ executable }} {{ input_file }} {{ output_file }}
if [ ! -z "{{ post_processor }}" ]
then
  {{ mpi_cmd }} -np {{ num_procs }} {{ post_processor }} --file={{ output_file }} {{ post_options }}
fi
