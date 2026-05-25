#include "test_utils.hpp"

#include "DType.hpp"

#include <string>

using Mimir::DType;

int main() {
    TASSERT_TRUE(Mimir::parse_dtype("float") == DType::F32);
    TASSERT_TRUE(Mimir::parse_dtype("f32") == DType::F32);
    TASSERT_TRUE(Mimir::parse_dtype("float32") == DType::F32);

    TASSERT_TRUE(Mimir::parse_dtype("double") == DType::F64);
    TASSERT_TRUE(Mimir::parse_dtype("f64") == DType::F64);
    TASSERT_TRUE(Mimir::parse_dtype("float64") == DType::F64);

    TASSERT_TRUE(Mimir::parse_dtype("float16") == DType::F16);
    TASSERT_TRUE(Mimir::parse_dtype("f16") == DType::F16);
    TASSERT_TRUE(Mimir::parse_dtype("fp16") == DType::F16);

    TASSERT_TRUE(Mimir::parse_dtype("bf16") == DType::BF16);
    TASSERT_TRUE(Mimir::parse_dtype("bfloat16") == DType::BF16);

    TASSERT_TRUE(Mimir::parse_dtype("i8") == DType::I8);
    TASSERT_TRUE(Mimir::parse_dtype("uint16") == DType::U16);
    TASSERT_TRUE(Mimir::parse_dtype("bool") == DType::BOOL);

    TASSERT_TRUE(Mimir::parse_dtype("not_a_dtype") == DType::UNKNOWN);

    TASSERT_TRUE(Mimir::dtype_size_bytes(DType::F16) == 2);
    TASSERT_TRUE(Mimir::dtype_size_bytes(DType::F32) == 4);
    TASSERT_TRUE(Mimir::dtype_size_bytes(DType::F64) == 8);
    TASSERT_TRUE(Mimir::dtype_size_bytes(DType::I64) == 8);
    TASSERT_TRUE(Mimir::dtype_size_bytes(DType::UNKNOWN) == 0);

    TASSERT_TRUE(std::string(Mimir::dtype_to_string(DType::F32)) == "float32");
    TASSERT_TRUE(std::string(Mimir::dtype_to_string(DType::BF16)) == "bfloat16");

    // safetensors codes
    TASSERT_TRUE(Mimir::parse_dtype_safetensors("F32") == DType::F32);
    TASSERT_TRUE(Mimir::parse_dtype_safetensors("I64") == DType::I64);
    TASSERT_TRUE(Mimir::parse_dtype_safetensors("BOOL") == DType::BOOL);

    return 0;
}
