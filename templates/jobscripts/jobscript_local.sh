/usr/bin/mpirun --bind-to none -np {{num_procs}} {{executable}} {{input_file}} {{output_file}}
if [ ! -z "{{ post_processor }}" ]
then
  /usr/bin/mpirun --bind-to none -np {{num_procs_post}} {{post_processor}} --file={{output_file}} {{post_options}} --output={{output_dir}}/{{post_file_prefix}}
fi
