# Exercise Rust through the real CLI, without rust-script or Cargo downloads.
set(test_dir "${CMAKE_CURRENT_BINARY_DIR}/rust conf smoke")
file(MAKE_DIRECTORY "${test_dir}")
set(script "${test_dir}/check.rs")
file(WRITE "${script}" [=[
assert_eq!(conf()?["marker"].as_i64(), Some(42));
assert!(conf_path()?.as_str().unwrap().ends_with("/config.json"));
assert!(conf_dir()?.as_str().unwrap().ends_with("/rust conf smoke"));
assert_eq!(arg()?[1].as_str(), Some("é漢字 | spaces"));
Model::create(&["basic_mlp".into(), Value::object(&[("input_dim",4.into()),("hidden_dim",8.into()),("output_dim",2.into())])])?;
Model::allocate_params(&[])?;
Model::init_weights(&["he".into(),42.into()])?;
assert!(Model::total_params(&[])?.as_u64().unwrap()>0);
println!("RUST_CONF_OK");
]=])
set(next_script "${test_dir}/next.rs")
file(WRITE "${next_script}" [=[
assert!(Model::total_params(&[])?.as_u64().unwrap()>0);
println!("RUST_PERSIST_OK");
]=])
set(block "{\"scripts\":[{\"script\":\"${script}\",\"args\":[\"é漢字 | spaces\"]},\"${next_script}\"]}")
foreach(mode root nested task)
    set(options)
    if(mode STREQUAL root)
        set(config "{\"marker\":42,\"rust\":${block}}")
    elseif(mode STREQUAL nested)
        set(config "{\"marker\":42,\"run\":{\"rust\":${block}}}")
    else()
        set(config "{\"marker\":42,\"tasks\":{\"check\":{\"rust\":${block}}}}")
        set(options --run check)
    endif()
    file(WRITE "${test_dir}/config.json" "${config}")
    execute_process(COMMAND "${MIMIR_BIN}" --conf "${test_dir}/config.json" ${options}
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT status EQUAL 0 OR NOT output MATCHES "RUST_CONF_OK" OR NOT output MATCHES "RUST_PERSIST_OK")
        message(FATAL_ERROR "Rust ${mode} config failed: ${status}\n${output}\n${error}")
    endif()
endforeach()
file(WRITE "${test_dir}/args.rs" [=[
assert_eq!(arg()?[1].as_str(), Some("--literal"));
assert_eq!(arg()?[2].as_str(), Some("é漢字 | spaces"));
assert_eq!(arg()?.as_array().unwrap().len(), 3);
]=])
execute_process(COMMAND "${MIMIR_BIN}" --rust "${test_dir}/args.rs" -- --literal "é漢字 | spaces"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Rust argument separator failed: ${status}\n${output}\n${error}")
endif()
file(WRITE "${script}" "return Err(mimir::Error(\"intentional\".into()));\n")
execute_process(COMMAND "${MIMIR_BIN}" --rust "${script}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(status EQUAL 0)
    message(FATAL_ERROR "Uncaught Rust error returned success")
endif()
file(WRITE "${script}" "invalid syntax !!!\n")
execute_process(COMMAND "${MIMIR_BIN}" --rust "${script}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(status EQUAL 0)
    message(FATAL_ERROR "Rust compile error returned success")
endif()
file(REMOVE_RECURSE "${test_dir}")
