
// OpenCL kernels for image processing operations

__kernel void convolution_2d(__global float* input,
                            __global float* output,
                            __global float* filter,
                            const int width,
                            const int height,
                            const int filter_size) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int half_filter = filter_size / 2;

    for (int fy = -half_filter; fy <= half_filter; fy++) {
        for (int fx = -half_filter; fx <= half_filter; fx++) {
            int px = clamp(x + fx, 0, width - 1);
            int py = clamp(y + fy, 0, height - 1);

            int filter_idx = (fy + half_filter) * filter_size + (fx + half_filter);
            sum += input[py * width + px] * filter[filter_idx];
        }
    }

    output[y * width + x] = sum;
}

__kernel void gaussian_blur(__global uchar4* input,
                           __global uchar4* output,
                           const int width,
                           const int height,
                           const float sigma) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    int radius = (int)(3.0f * sigma);

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int px = clamp(x + dx, 0, width - 1);
            int py = clamp(y + dy, 0, height - 1);

            float distance = sqrt((float)(dx*dx + dy*dy));
            float weight = exp(-(distance * distance) / (2.0f * sigma * sigma));

            uchar4 pixel = input[py * width + px];
            sum += convert_float4(pixel) * weight;
            weight_sum += weight;
        }
    }

    sum /= weight_sum;
    output[y * width + x] = convert_uchar4(sum);
}

__kernel void resize_bilinear(__global uchar4* input,
                             __global uchar4* output,
                             const int src_width,
                             const int src_height,
                             const int dst_width,
                             const int dst_height) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= dst_width || y >= dst_height) return;

    float src_x = ((float)x / dst_width) * src_width;
    float src_y = ((float)y / dst_height) * src_height;

    int x1 = (int)floor(src_x);
    int y1 = (int)floor(src_y);
    int x2 = min(x1 + 1, src_width - 1);
    int y2 = min(y1 + 1, src_height - 1);

    float dx = src_x - x1;
    float dy = src_y - y1;

    uchar4 p11 = input[y1 * src_width + x1];
    uchar4 p12 = input[y2 * src_width + x1];
    uchar4 p21 = input[y1 * src_width + x2];
    uchar4 p22 = input[y2 * src_width + x2];

    float4 f11 = convert_float4(p11);
    float4 f12 = convert_float4(p12);
    float4 f21 = convert_float4(p21);
    float4 f22 = convert_float4(p22);

    float4 result = f11 * (1-dx) * (1-dy) + 
                   f21 * dx * (1-dy) + 
                   f12 * (1-dx) * dy + 
                   f22 * dx * dy;

    output[y * dst_width + x] = convert_uchar4(result);
}
