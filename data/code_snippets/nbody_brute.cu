#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define CUDA_CHECK(call)                                                       \
{                                                                              \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
        printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                   \
        printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
        exit(1);                                                               \
    }                                                                          \
}

#define SOFTENING 1e-9f

struct Particles {
    float *x_in, *x_out, *y_in, *y_out;
    float *vx, *vy;
    int num;
};

float random_float(float min, float max) {
    static mt19937 rng(random_device{}());
    uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

__global__ void nbody_kern(Particles particles) {
    float* x_in = particles.x_in;
    float* x_out = particles.x_out;
    float* y_in = particles.y_in;
    float* y_out = particles.y_out;
    float* vx = particles.vx;
    float* vy = particles.vy;
    int numParticles = particles.num;

    int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (thread_id < numParticles) {
        float p_x = x_in[thread_id];
        float p_y = y_in[thread_id];
        float p_vx = vx[thread_id];
        float p_vy = vy[thread_id];

        float Fx = 0.0f; float Fy = 0.0f;
        float dt = 0.01f;

        for (int i = 0; i < numParticles; i++) {
            if (thread_id == i)
                continue;
            
            float dx = x_in[i] - p_x;
            float dy = y_in[i] - p_y;
            float distSqr = dx*dx + dy*dy + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3;
        }

        p_vx += dt*Fx; p_vy += dt*Fy;
        p_x += p_vx*dt; p_y += p_vy*dt;
        vx[thread_id]    = p_vx; vy[thread_id]    = p_vy;
        x_out[thread_id] = p_x; y_out[thread_id] = p_y;
    }
}

__host__ void initialize_states(float *f, int n, float min, float max) {
    for (int i = 0; i < n; i++)
        f[i] = random_float(min, max);
}

__host__ void initialize_host_particles(Particles *h_particles, size_t bytes) {
    h_particles->x_in = (float*) malloc(bytes);
    h_particles->x_out = (float*) malloc(bytes);
    h_particles->y_in = (float*) malloc(bytes);
    h_particles->y_out = (float*) malloc(bytes);
    h_particles->vx = (float*) malloc(bytes);
    h_particles->vy = (float*) malloc(bytes);

    initialize_states(h_particles->x_in, h_particles->num, -50, 50);
    initialize_states(h_particles->y_in, h_particles->num, -50, 50);
    initialize_states(h_particles->vx, h_particles->num, -5, 5);
    initialize_states(h_particles->vy, h_particles->num, -5, 5);
}

__host__ void save_particles(Particles *h_particles) {
    
}

int main(const int argc, const char** argv) {
    int NUM_PARTICLES = 3000;
    float L_MIN = -50.0f;
    float L_MAX = 50.0f;
    float V_MIN = -2.0f;
    float V_MAX = 2.0f;

    if (argc > 1)
        NUM_PARTICLES      = atoi(argv[1]);

    const size_t bytes = NUM_PARTICLES * sizeof(float);

    float *h_x, *h_y, *h_vx, *h_vy;
    h_x = (float*) malloc(bytes);
    h_y = (float*) malloc(bytes);
    h_vx = (float*) malloc(bytes);
    h_vy = (float*) malloc(bytes);

    initialize_states(h_x, NUM_PARTICLES, L_MIN, L_MAX);
    initialize_states(h_y, NUM_PARTICLES, L_MIN, L_MAX);
    initialize_states(h_vx, NUM_PARTICLES, V_MIN, V_MAX);
    initialize_states(h_vy, NUM_PARTICLES, V_MIN, V_MAX);

    Particles d_particles;
    float *d_x_in, *d_x_out, *d_y_in, *d_y_out, *d_vx, *d_vy;
    CUDA_CHECK(cudaMalloc(&d_x_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_x_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_y_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_y_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));

    CUDA_CHECK(cudaMemcpy(d_x_in, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_in, h_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice));

    d_particles.num = NUM_PARTICLES;
    d_particles.x_in = d_x_in;
    d_particles.x_out = d_x_out;
    d_particles.y_in = d_y_in;
    d_particles.y_out = d_y_out;
    d_particles.vx = d_vx;
    d_particles.vy = d_vy;

    int block_size = 128;
    int grid_size  = (NUM_PARTICLES + block_size - 1) / block_size;

    float dt = 0.01f;
    float T = 20;
    int numSteps = static_cast<int>(T / dt);

    FILE* f = fopen("nbody.bin", "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }
    fwrite(&NUM_PARTICLES, sizeof(int), 1, f);
    fwrite(&numSteps, sizeof(int), 1, f);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float elapsed_ms = 0.0f;

    for (float t = 0.0f; t < T; t += dt) {
        CUDA_CHECK(cudaEventRecord(start));

        nbody_kern<<<grid_size, block_size>>>(d_particles);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_x, d_x_out, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y, d_y_out, bytes, cudaMemcpyDeviceToHost));

        float step_ms;
        CUDA_CHECK(cudaEventElapsedTime(&step_ms, start, stop));
        elapsed_ms += step_ms;

        size_t wrote;
        wrote = fwrite(h_x, sizeof(float), NUM_PARTICLES, f);
        if (wrote != (size_t)NUM_PARTICLES) {
            perror("fwrite x");
            return 1;
        }
        wrote = fwrite(h_y, sizeof(float), NUM_PARTICLES, f);
        if (wrote != (size_t)NUM_PARTICLES) {
            perror("fwrite y");
            return 1;
        }

        float* tmp;
        tmp = d_x_in;  d_x_in  = d_x_out;  d_x_out  = tmp;
        tmp = d_y_in;  d_y_in  = d_y_out;  d_y_out  = tmp;

        d_particles.x_in  = d_x_in;
        d_particles.x_out = d_x_out;
        d_particles.y_in  = d_y_in;
        d_particles.y_out = d_y_out;
    }

    CUDA_CHECK(cudaFree(d_x_in));
    CUDA_CHECK(cudaFree(d_y_in));
    CUDA_CHECK(cudaFree(d_x_out));
    CUDA_CHECK(cudaFree(d_y_out));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));

    free(h_x);
    free(h_y);
    free(h_vx);
    free(h_vy);

    fclose(f);

    CUDA_CHECK(cudaDeviceReset());
    printf("Total kernel time: %.3f ms\n", elapsed_ms);
    printf("Avg kernel time per step: %.3f ms\n", elapsed_ms / numSteps);

    return 0;
}