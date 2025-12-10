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

struct Node {
    float x_min, x_max, y_min, y_max;
    float mass;
    float x_com;
    float y_com;
    int   child[4];  
    int   start;     
    int   count;     
};

float random_float(float min, float max) {
    static mt19937 rng(random_device{}());
    uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

__device__ float2 bh_force_for_particle(int p, float px, float py,
    const Node*  __restrict__ nodes,
    const int*   __restrict__ pid,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float theta2)
{
    float Fx = 0.0f, Fy = 0.0f;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // root = node 0

    while (sp > 0) {
        int idx = stack[--sp];
        const Node& n = nodes[idx];

        if (n.count == 0) continue;

        bool is_leaf =
            (n.child[0] < 0) &&
            (n.child[1] < 0) &&
            (n.child[2] < 0) &&
            (n.child[3] < 0);

        float dx_box = n.x_max - n.x_min;
        float dy_box = n.y_max - n.y_min;
        float s = fmaxf(dx_box, dy_box);

        float dx = n.x_com - px;
        float dy = n.y_com - py;
        float r2 = dx*dx + dy*dy + SOFTENING;

        if (is_leaf) {
            for (int i = 0; i < n.count; ++i) {
                int pj = pid[n.start + i];
                if (pj == p) continue;

                float qx = x[pj];
                float qy = y[pj];
                float dxp = qx - px;
                float dyp = qy - py;
                float r2p = dxp*dxp + dyp*dyp + SOFTENING;

                float invR  = rsqrtf(r2p);
                float invR3 = invR * invR * invR;

                Fx += dxp * invR3;
                Fy += dyp * invR3;
            }
        } else if ((s*s) / r2 < theta2) {
            float invR  = rsqrtf(r2);
            float invR3 = invR * invR * invR;
            float m     = n.mass;

            Fx += m * dx * invR3;
            Fy += m * dy * invR3;
        } else {
            for (int c = 0; c < 4; ++c) {
                int ci = n.child[c];
                if (ci >= 0) {
                    stack[sp++] = ci;
                }
            }
        }
    }

    return make_float2(Fx, Fy);
}


__global__ void nbody_bh_kern(Particles particles, const Node* __restrict__ nodes, const int*  __restrict__ pid, float theta, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= particles.num) return;

    float* x_in  = particles.x_in;
    float* x_out = particles.x_out;
    float* y_in  = particles.y_in;
    float* y_out = particles.y_out;
    float* vx    = particles.vx;
    float* vy    = particles.vy;

    float px  = x_in[idx];
    float py  = y_in[idx];
    float pvx = vx[idx];
    float pvy = vy[idx];

    float2 F = bh_force_for_particle(idx, px, py, nodes, pid, x_in, y_in, theta*theta);
    float Fx = F.x;
    float Fy = F.y;

    pvx += dt * Fx;
    pvy += dt * Fy;
    px  += dt * pvx;
    py  += dt * pvy;

    vx[idx]    = pvx;
    vy[idx]    = pvy;
    x_out[idx] = px;
    y_out[idx] = py;
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

int create_node(vector<Node>& nodes) {
    nodes.push_back(Node{});
    Node& n = nodes.back();
    n.mass = 0.0f;
    n.x_com = n.y_com = 0.0f;
    n.start = n.count = 0;
    for (int i = 0; i < 4; ++i) n.child[i] = -1;
    return (int)nodes.size() - 1;
}

void build_node(int node_idx, vector<Node>& nodes, vector<int>& pid, float* x, float* y, int start, int count, int max_leaf = 8) {
    Node& n = nodes[node_idx];
    n.start = start;
    n.count = count;

    float x_min = 1e30f, x_max = -1e30f;
    float y_min = 1e30f, y_max = -1e30f;
    float mass = 0.0f;
    float x_sum = 0.0f, y_sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        int p = pid[start + i];
        float px = x[p];
        float py = y[p];
        if (px < x_min) x_min = px;
        if (px > x_max) x_max = px;
        if (py < y_min) y_min = py;
        if (py > y_max) y_max = py;
        mass  += 1.0f;
        x_sum += px;
        y_sum += py;
    }
    n.x_min = x_min;
    n.x_max = x_max;
    n.y_min = y_min;
    n.y_max = y_max;
    n.mass  = mass;
    n.x_com = (mass > 0.0f) ? (x_sum / mass) : 0.0f;
    n.y_com = (mass > 0.0f) ? (y_sum / mass) : 0.0f;

    if (count <= max_leaf)
        return;

    float x_mid = 0.5f * (x_min + x_max);
    float y_mid = 0.5f * (y_min + y_max);

    auto quadrant = [&](float px, float py) {
        if (px > x_mid) {
            return (py > y_mid) ? 0 : 3; // NE, SE
        } else {
            return (py > y_mid) ? 1 : 2; // NW, SW
        }
    };

    int q_count[4] = {0, 0, 0, 0};
    for (int i = 0; i < count; ++i) {
        int p = pid[start + i];
        int q = quadrant(x[p], y[p]);
        q_count[q]++;
    }

    int q_offset[4];
    int acc = start;
    for (int q = 0; q < 4; ++q) {
        q_offset[q] = acc;
        acc += q_count[q];
    }

    int q_written[4] = {0,0,0,0};
    std::vector<int> tmp(count);
    for (int i = 0; i < count; ++i) {
        int p = pid[start + i];
        int q = quadrant(x[p], y[p]);
        int dst = q_offset[q] + q_written[q]++;
        tmp[dst - start] = p;
    }
    for (int i = 0; i < count; ++i) {
        pid[start + i] = tmp[i];
    }

    for (int q = 0; q < 4; ++q) {
        if (q_count[q] == 0) {
            nodes[node_idx].child[q] = -1;
            continue;
        }
        int child_idx = create_node(nodes);
        nodes[node_idx].child[q] = child_idx;
        build_node(child_idx, nodes, pid,
                   x, y,
                   q_offset[q], q_count[q],
                   max_leaf);
    }
}


int main(const int argc, const char** argv) {
    int NUM_PARTICLES = 3000;
    float L_MIN = -50.0f;
    float L_MAX = 50.0f;
    float V_MIN = -2.0f;
    float V_MAX = 2.0f;
    float THETA = 0.5f;

    if (argc > 2) {
        NUM_PARTICLES = atoi(argv[1]);
        THETA = atof(argv[2]);
    }

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

    FILE* f = fopen("nbody_bh.bin", "wb");
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
        vector<Node> h_nodes;
        vector<int>  h_pid(NUM_PARTICLES);
        for (int i = 0; i < NUM_PARTICLES; ++i) h_pid[i] = i;

        int root_idx = create_node(h_nodes);
        build_node(root_idx, h_nodes, h_pid, h_x, h_y, 0, NUM_PARTICLES);
        int numNodes = (int)h_nodes.size();

        Node* d_nodes;
        int*  d_pid;
        CUDA_CHECK(cudaMalloc(&d_nodes, numNodes * sizeof(Node)));
        CUDA_CHECK(cudaMalloc(&d_pid, NUM_PARTICLES * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes.data(), numNodes * sizeof(Node), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pid, h_pid.data(), NUM_PARTICLES * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start));

        nbody_bh_kern<<<grid_size, block_size>>>(d_particles, d_nodes, d_pid, THETA, dt);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_x, d_x_out, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y, d_y_out, bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_nodes));
        CUDA_CHECK(cudaFree(d_pid));

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