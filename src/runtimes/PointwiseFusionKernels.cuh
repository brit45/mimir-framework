#pragma once
// Shared CUDA/HIP source; also translated to OpenCL and SPIR-V by the
// portable generator. Program entries are (unary opcode, LeakyReLU alpha).
__device__ static inline float fusion_value(float x,int op,float alpha) {
    switch (op) {
        case 0:
            return x > 0.0f ? x : 0.0f;
        case 1:
            return x > 0.0f ? x : alpha * x;
        case 2:
            return 1.0f / (1.0f + expf(-x));
        case 3:
            return tanhf(x);
        case 4: {
            const float s = 1.0f / (1.0f + expf(-x));
            return x * s;
        }
        case 5: {
            const float c = 0.7978845608f;
            const float x3 = x * x * x;
            return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
        }
        case 6:
            return x > 20.0f ? x : log1pf(expf(x));
        case 7: {
            const float sp = x > 20.0f ? x : log1pf(expf(x));
            return x * tanhf(sp);
        }
        case 8: {
            const float hs = (x + 3.0f) / 6.0f;
            return fminf(1.0f, fmaxf(0.0f, hs));
        }
        case 9: {
            const float hs = fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
            return x * hs;
        }
        default:
            return x;
    }
}
__device__ static inline float fusion_derivative(float x,int op,float alpha) {
    float d = 1.0f;

    switch (op) {
        case 0:
            d = x > 0.0f ? 1.0f : 0.0f;
            break;
        case 1:
            d = x > 0.0f ? 1.0f : alpha;
            break;
        case 2: {
            const float s = 1.0f / (1.0f + expf(-x));
            d = s * (1.0f - s);
            break;
        }
        case 3: {
            const float t = tanhf(x);
            d = 1.0f - t * t;
            break;
        }
        case 4: {
            const float s = 1.0f / (1.0f + expf(-x));
            d = s + x * s * (1.0f - s);
            break;
        }
        case 5: {
            const float c = 0.7978845608f;
            const float x2 = x * x;
            const float x3 = x2 * x;
            const float u = c * (x + 0.044715f * x3);
            const float t = tanhf(u);
            const float sech2 = 1.0f - t * t;
            const float du = c * (1.0f + 0.134145f * x2);
            d = 0.5f * (1.0f + t) + 0.5f * x * sech2 * du;
            break;
        }
        case 6:
            d = 1.0f / (1.0f + expf(-x));
            break;
        case 7: {
            const float sp = x > 20.0f ? x : log1pf(expf(x));
            const float t = tanhf(sp);
            const float s = 1.0f / (1.0f + expf(-x));
            d = t + x * s * (1.0f - t * t);
            break;
        }
        case 8:
            d = (x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f;
            break;
        case 9:
            if (x <= -3.0f) d = 0.0f;
            else if (x >= 3.0f) d = 1.0f;
            else d = x / 3.0f + 0.5f;
            break;
        default:
            d = 1.0f;
            break;
    }

    return d;
}
__global__ static void pointwise_fusion_forward(const float* input,const float* program,float* output,int n,int stages,bool retain) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float value=input[i];
    for(int j=0;j<stages;++j) {
        value=fusion_value(value,int(program[2*j]),program[2*j+1]);
        if(retain)output[j*n+i]=value;
    }
    if(!retain)output[i]=value;
}
__global__ static void pointwise_fusion_backward(const float* gradient,const float* snapshots,const float* program,float* output,int n,int stages) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float value=gradient[i];
    for(int j=stages-1;j>=0;--j) {
        float d=fusion_derivative(snapshots[j*n+i],int(program[2*j]),program[2*j+1]);
        value=value*d;
    }
    output[i]=value;
}
