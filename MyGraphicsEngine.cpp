#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <limits>
#include <random>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <chrono>

#define WIDTH 512
#define HEIGHT 512
#define M_PI 3.14159265358
#define MAX_LIGHT_INTENSITY 1e10
#define GAMMA 2.2
#define EPSILON 1e-6
#define DEFAULT_MAX_RECURSION_DEPTH 4
#define NB_RAY 16
#define DEFAULT_STD_ANTIALIASING 0.5

#ifdef _OPENMP
    #include <omp.h>
    static const int num_cores{ omp_get_num_procs() };
#else
    static const int num_cores{ 1 };
#endif

static std::vector<std::default_random_engine> engines(num_cores);
static std::uniform_real_distribution<double> uniform(0, 1);

static inline double sqr(double x) { return x * x; }

static unsigned char color_correction(double num) {

    num = pow(num, 1/GAMMA); // correction gamma
    if (num > 255) return 255; // clamping supérieur
    else if (num < 0) return 0; // clamping inférieur
    else return static_cast<unsigned char>(num); // conversion
};

static void boxMuller(double& dx, double& dy, double stdev = DEFAULT_STD_ANTIALIASING) {
    #ifdef _OPENMP
        int thread_id{ omp_get_thread_num() };
    #else
        int thread_id{ 0 };
    #endif
    double r1 = uniform(engines[thread_id]);
    double r2 = uniform(engines[thread_id]);
    double R = sqrt(-2 * log(r1));
    dx = R * cos(2 * M_PI * r2) * stdev;
    dy = R * sin(2 * M_PI * r2) * stdev;
}

class Vector {
public:
    explicit Vector(double x = 0, double y = 0, double z = 0) {
        coord[0] = x;
        coord[1] = y;
        coord[2] = z;
    }
    double& operator[](int i) { return coord[i]; }
    double operator[](int i) const { return coord[i]; }

    Vector& operator+=(const Vector& v) {
        coord[0] += v[0];
        coord[1] += v[1];
        coord[2] += v[2];
        return *this;
    }

    double norm2() const {
        return sqr(coord[0]) + sqr(coord[1]) + sqr(coord[2]);
    }

    void normalize() {
        double norm{ sqrt(norm2()) };
        coord[0] /= norm;
        coord[1] /= norm;
        coord[2] /= norm;
    };

    double coord[3];
};

static Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
static Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
static Vector operator*(const Vector& a, double b) {
    return Vector(a[0]*b, a[1]*b, a[2]*b);
}
static Vector operator*(double a, const Vector& b) {
    return Vector(a*b[0], a*b[1], a*b[2]);
}

static Vector operator*(const Vector& a, const Vector& b) {
    return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

static Vector operator/(const Vector& a, double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}

static double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static Vector cross(const Vector& a, const Vector& b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b [0]);
}

class TriangleIndices {
public:
    TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group) {
    };
    int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
    int uvi, uvj, uvk;  // indices within the uv coordinates array
    int ni, nj, nk;  // indices within the normals array
    int group;       // face group
};

static Vector random_cos(const Vector& N) {

    #ifdef _OPENMP
        int thread_id{ omp_get_thread_num() };
    #else
        int thread_id{ 0 };
    #endif

    double r1{ uniform(engines[thread_id]) };
    double r2{ uniform(engines[thread_id]) };

    double s{ sqrt(1 - r2) };
    double x{ cos(2 * M_PI * r1) * s };
    double y{ sin(2 * M_PI * r1) * s };
    double z{ sqrt(r2) };

    Vector T(0, 0, 0);
    if (abs(N[0]) < abs(N[1]) && abs(N[0]) < abs(N[2])) {
        T[1] = -N[2];
        T[2] = N[1];
    }
    else if (abs(N[1]) < abs(N[0]) && abs(N[1]) < abs(N[2])) {
        T[0] = -N[2];
        T[2] = N[0];
    }
    else {
        T[0] = -N[1];
        T[1] = N[0];
    }
    T.normalize();

    Vector T2{ cross(T, N) };

    return x * T + y * T2 + z * N;
}

class Ray {
public:
    
    Ray(const Vector& origin, const Vector& direction): origin(origin), direction(direction) {};

    Vector origin, direction;
};

class Geometry {
public:
    Geometry(const Vector& albedo = Vector(1, 1, 1),
        bool isMirror = false,
        bool isTransparent = false,
        double refraction_index = 1.0):
        albedo(albedo),
        isMirror(isMirror),
        isTransparent(isTransparent),
        refraction_index(refraction_index) {}

    virtual bool intersect(const Ray& ray, Vector& intersection_point, Vector& intersection_normal, double& t, Vector& albedo) const = 0;

    Vector albedo;
    bool isMirror;
    bool isTransparent;
    double refraction_index;
};

class BoundingBox {
public:

    BoundingBox(const Vector& m = Vector(0, 0, 0), const Vector& M = Vector(0, 0, 0)): m(m), M(M) {}

    bool intersect(const Ray& ray) const {

        // cette fonction a directement été copiée au tableau pendant la séance de code en cours
        // (comme j'espère l'avoir montré dans le partiel, je l'ai comprise depuis)

        double P1x = (m[0] - ray.origin[0]) / ray.direction[0];
        double P2x = (M[0] - ray.origin[0]) / ray.direction[0];
        double xmin = std::min(P1x, P2x);
        double xmax = P1x + P2x - xmin;

        double P1y = (m[1] - ray.origin[1]) / ray.direction[1];
        double P2y = (M[1] - ray.origin[1]) / ray.direction[1];
        double ymin = std::min(P1y, P2y);
        double ymax = P1y + P2y - ymin;

        double P1z = (m[2] - ray.origin[2]) / ray.direction[2];
        double P2z = (M[2] - ray.origin[2]) / ray.direction[2];
        double zmin = std::min(P1z, P2z);
        double zmax = P1z + P2z - zmin;
        
        double max_of_min = std::max(xmin, std::max(ymin, zmin));
        double min_of_max = std::min(xmax, std::min(ymax, zmax));

        if (min_of_max < 0) return false;

        return (min_of_max > max_of_min);
    }

    Vector m;
    Vector M;
};

class BVH {
public:

    BVH* leftChild;
    BVH* rightChild;
    BoundingBox bbox;
    int start, end;
};

class TriangleMesh : public Geometry {
public:
    ~TriangleMesh() {}
    TriangleMesh() {};

    void readOBJ(const char* obj) {

        char matfile[255];
        char grp[255];

        FILE* f;
        f = fopen(obj, "r");
        int curGroup = -1;
        while (!feof(f)) {
            char line[255];
            if (!fgets(line, 255, f)) break;

            std::string linetrim(line);
            linetrim.erase(linetrim.find_last_not_of(" \r\t") + 1);
            strcpy(line, linetrim.c_str());

            if (line[0] == 'u' && line[1] == 's') {
                sscanf(line, "usemtl %[^\n]\n", grp);
                curGroup++;
            }

            if (line[0] == 'v' && line[1] == ' ') {
                Vector vec;

                Vector col;
                if (sscanf(line, "v %lf %lf %lf %lf %lf %lf\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6) {
                    col[0] = std::min(1., std::max(0., col[0]));
                    col[1] = std::min(1., std::max(0., col[1]));
                    col[2] = std::min(1., std::max(0., col[2]));

                    vertices.push_back(vec);
                    vertexcolors.push_back(col);

                }
                else {
                    sscanf(line, "v %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                    vertices.push_back(vec);
                }
            }
            if (line[0] == 'v' && line[1] == 'n') {
                Vector vec;
                sscanf(line, "vn %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                normals.push_back(vec);
            }
            if (line[0] == 'v' && line[1] == 't') {
                Vector vec;
                sscanf(line, "vt %lf %lf\n", &vec[0], &vec[1]);
                uvs.push_back(vec);
            }
            if (line[0] == 'f') {
                TriangleIndices t;
                int i0, i1, i2, i3;
                int j0, j1, j2, j3;
                int k0, k1, k2, k3;
                int nn;
                t.group = curGroup;

                char* consumedline = line + 1;
                int offset;

                nn = sscanf(consumedline, "%u/%u/%u %u/%u/%u %u/%u/%u%n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2, &offset);
                if (nn == 9) {
                    if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                    if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                    if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                    if (j0 < 0) t.uvi = uvs.size() + j0; else   t.uvi = j0 - 1;
                    if (j1 < 0) t.uvj = uvs.size() + j1; else   t.uvj = j1 - 1;
                    if (j2 < 0) t.uvk = uvs.size() + j2; else   t.uvk = j2 - 1;
                    if (k0 < 0) t.ni = normals.size() + k0; else    t.ni = k0 - 1;
                    if (k1 < 0) t.nj = normals.size() + k1; else    t.nj = k1 - 1;
                    if (k2 < 0) t.nk = normals.size() + k2; else    t.nk = k2 - 1;
                    indices.push_back(t);
                }
                else {
                    nn = sscanf(consumedline, "%u/%u %u/%u %u/%u%n", &i0, &j0, &i1, &j1, &i2, &j2, &offset);
                    if (nn == 6) {
                        if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                        if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                        if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                        if (j0 < 0) t.uvi = uvs.size() + j0; else   t.uvi = j0 - 1;
                        if (j1 < 0) t.uvj = uvs.size() + j1; else   t.uvj = j1 - 1;
                        if (j2 < 0) t.uvk = uvs.size() + j2; else   t.uvk = j2 - 1;
                        indices.push_back(t);
                    }
                    else {
                        nn = sscanf(consumedline, "%u %u %u%n", &i0, &i1, &i2, &offset);
                        if (nn == 3) {
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                            indices.push_back(t);
                        }
                        else {
                            nn = sscanf(consumedline, "%u//%u %u//%u %u//%u%n", &i0, &k0, &i1, &k1, &i2, &k2, &offset);
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                            if (k0 < 0) t.ni = normals.size() + k0; else    t.ni = k0 - 1;
                            if (k1 < 0) t.nj = normals.size() + k1; else    t.nj = k1 - 1;
                            if (k2 < 0) t.nk = normals.size() + k2; else    t.nk = k2 - 1;
                            indices.push_back(t);
                        }
                    }
                }

                consumedline = consumedline + offset;

                while (true) {
                    if (consumedline[0] == '\n') break;
                    if (consumedline[0] == '\0') break;
                    nn = sscanf(consumedline, "%u/%u/%u%n", &i3, &j3, &k3, &offset);
                    TriangleIndices t2;
                    t2.group = curGroup;
                    if (nn == 3) {
                        if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                        if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                        if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                        if (j0 < 0) t2.uvi = uvs.size() + j0; else  t2.uvi = j0 - 1;
                        if (j2 < 0) t2.uvj = uvs.size() + j2; else  t2.uvj = j2 - 1;
                        if (j3 < 0) t2.uvk = uvs.size() + j3; else  t2.uvk = j3 - 1;
                        if (k0 < 0) t2.ni = normals.size() + k0; else   t2.ni = k0 - 1;
                        if (k2 < 0) t2.nj = normals.size() + k2; else   t2.nj = k2 - 1;
                        if (k3 < 0) t2.nk = normals.size() + k3; else   t2.nk = k3 - 1;
                        indices.push_back(t2);
                        consumedline = consumedline + offset;
                        i2 = i3;
                        j2 = j3;
                        k2 = k3;
                    }
                    else {
                        nn = sscanf(consumedline, "%u/%u%n", &i3, &j3, &offset);
                        if (nn == 2) {
                            if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                            if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                            if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                            if (j0 < 0) t2.uvi = uvs.size() + j0; else  t2.uvi = j0 - 1;
                            if (j2 < 0) t2.uvj = uvs.size() + j2; else  t2.uvj = j2 - 1;
                            if (j3 < 0) t2.uvk = uvs.size() + j3; else  t2.uvk = j3 - 1;
                            consumedline = consumedline + offset;
                            i2 = i3;
                            j2 = j3;
                            indices.push_back(t2);
                        }
                        else {
                            nn = sscanf(consumedline, "%u//%u%n", &i3, &k3, &offset);
                            if (nn == 2) {
                                if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                                if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                                if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                                if (k0 < 0) t2.ni = normals.size() + k0; else   t2.ni = k0 - 1;
                                if (k2 < 0) t2.nj = normals.size() + k2; else   t2.nj = k2 - 1;
                                if (k3 < 0) t2.nk = normals.size() + k3; else   t2.nk = k3 - 1;
                                consumedline = consumedline + offset;
                                i2 = i3;
                                k2 = k3;
                                indices.push_back(t2);
                            }
                            else {
                                nn = sscanf(consumedline, "%u%n", &i3, &offset);
                                if (nn == 1) {
                                    if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                                    if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                                    if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                                    consumedline = consumedline + offset;
                                    i2 = i3;
                                    indices.push_back(t2);
                                }
                                else {
                                    consumedline = consumedline + 1;
                                }
                            }
                        }
                    }
                }

            }

        }
        fclose(f);
    }

    void transform(double scale, const Vector& translate) {
        for (int i{ 0 }; i < vertices.size(); i++) {
            vertices[i] = scale * vertices[i];
            vertices[i] = vertices[i] + translate;
        }
    }

    bool intersect(const Ray& ray, Vector& intersection_point, Vector& intersection_normal, double& smallest_t, Vector& albedo) const {

        bool hasIntersected{ false };
        smallest_t = std::numeric_limits<double>::max();

        if (!bbox.intersect(ray)) return false;

        std::vector<const BVH*> listBVH;
        listBVH.push_back(&root);

        while (!listBVH.empty()) {

            const BVH* current = listBVH.back();
            listBVH.pop_back();

            if (current->leftChild) {

                if (current->leftChild->bbox.intersect(ray)) {
                    listBVH.push_back(current->leftChild);
                }
                if (current->rightChild->bbox.intersect(ray)) {
                    listBVH.push_back(current->rightChild);
                }
            }
            else {

                for (int i{ current->start }; i < current->end; i++) {

                    Vector A{ vertices[indices[i].vtxi] };
                    Vector B{ vertices[indices[i].vtxj] };
                    Vector C{ vertices[indices[i].vtxk] };

                    Vector e1{ B - A };
                    Vector e2{ C - A };

                    Vector N{ cross(e1, e2) };
                    double inv_dot_prod{ 1 / dot(ray.direction, N) };
                    Vector cross_prod{ cross(ray.origin - A, ray.direction) };

                    double beta{ -dot(e2, cross_prod) * inv_dot_prod };
                    double gamma{ dot(e1, cross_prod) * inv_dot_prod };
                    double alpha{ 1 - beta - gamma };

                    if (alpha < 0) continue;
                    if (beta > 1) continue;
                    if (beta < 0) continue;
                    if (gamma > 1) continue;
                    if (gamma < 0) continue;

                    double t{ -dot(ray.origin - A, N) * inv_dot_prod };

                    if (t < 0) continue;

                    hasIntersected = true;

                    if (t > smallest_t) continue;

                    smallest_t = t;
                    //intersection_normal = N;
                    intersection_normal = (alpha * normals[indices[i].ni] + beta * normals[indices[i].nj] + gamma * normals[indices[i].nk]) / 3;
                    intersection_normal.normalize();
                    intersection_point = (ray.origin + EPSILON * intersection_normal) + ray.direction * smallest_t;

                    if (textures.size() != 0) {
                        Vector uv{ alpha * uvs[indices[i].uvi] + beta * uvs[indices[i].uvj] + gamma * uvs[indices[i].uvk] };
                        int w{ texW[indices[i].group] };
                        int h{ texH[indices[i].group] };
                        int uvx = fmod(uv[0] + 10000, 1.) * w;
                        int uvy = (1 - fmod(uv[1] + 10000, 1.)) * h;
                        
                        albedo[0] = textures[indices[i].group][(uvy * w + uvx) * 3 + 0];
                        albedo[1] = textures[indices[i].group][(uvy * w + uvx) * 3 + 1];
                        albedo[2] = textures[indices[i].group][(uvy * w + uvx) * 3 + 2];
                    }
                    else {
                        albedo = Vector(1, 1, 1); // white default color
                    }
                }
            }
        }

        return hasIntersected;

    }


    BoundingBox compute_bbox(int start_triangle, int end_triangle) {

        Vector M(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
        Vector m(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());

        for (int triangle_idx{ start_triangle }; triangle_idx < end_triangle; triangle_idx++) {

            int vtx_indices[3]{ indices[triangle_idx].vtxi, indices[triangle_idx].vtxj, indices[triangle_idx].vtxk };

            for (int local_vtx_idx{ 0 }; local_vtx_idx < 3; local_vtx_idx++) {

                int vtx_idx{ vtx_indices[local_vtx_idx] };

                if (vertices[vtx_idx][0] < m[0]) m[0] = vertices[vtx_idx][0]; // min selon x
                if (vertices[vtx_idx][1] < m[1]) m[1] = vertices[vtx_idx][1]; // min selon y
                if (vertices[vtx_idx][2] < m[2]) m[2] = vertices[vtx_idx][2]; // min selon z
                if (vertices[vtx_idx][0] > M[0]) M[0] = vertices[vtx_idx][0]; // max selon x
                if (vertices[vtx_idx][1] > M[1]) M[1] = vertices[vtx_idx][1]; // max selon y
                if (vertices[vtx_idx][2] > M[2]) M[2] = vertices[vtx_idx][2]; // max selon z
            }
        }

        return BoundingBox(m, M);
    }

    void build_BVH(BVH* bvh, int start, int end) {

        bvh->start = start;
        bvh->end = end;
        bvh->bbox = compute_bbox(start, end);
        bvh->leftChild = NULL;
        bvh->rightChild = NULL;

        Vector diag{ bvh->bbox.M - bvh->bbox.m };
        int axis{ 2 };
        if (diag[0] >= diag[1] && diag[0] >= diag[2]) axis = 0;
        if (diag[1] >= diag[0] && diag[1] >= diag[2]) axis = 1;
        double middle{ (bvh->bbox.M[axis] + bvh->bbox.m[axis]) / 2 };

        int pivot{ start };
        for (int i{ start }; i < end; i++) {
            double baryTriangleAxis{ (vertices[indices[i].vtxi][axis] + vertices[indices[i].vtxj][axis] + vertices[indices[i].vtxk][axis]) / 3 };
            if (baryTriangleAxis < middle) {
                std::swap(indices[i], indices[pivot]);
                pivot++;
            }
        }

        if (end - start <= 4) return;
        if (pivot - start == 0) return;
        if (end - pivot == 0) return;

        bvh->leftChild = new BVH;
        bvh->rightChild = new BVH;
        build_BVH(bvh->leftChild, start, pivot);
        build_BVH(bvh->rightChild, pivot, end);
    }

    void add_texture(const char* filename) {

        int w, h, c;
        unsigned char* tex = stbi_load(filename, &w, &h, &c, 3);
        texW.push_back(w);
        texH.push_back(h);
        std::vector<double> texture(w * h * 3);
        for (int i{ 0 }; i < w * h * 3; i++) {
            texture[i] = pow(tex[i]/255.0, GAMMA);
        }
        textures.push_back(texture);
    }

    std::vector<TriangleIndices> indices;
    std::vector<Vector> vertices;
    std::vector<Vector> normals;
    std::vector<Vector> uvs;
    std::vector<Vector> vertexcolors;

    BoundingBox bbox;
    BVH root;

    std::vector<std::vector<double>> textures;
    std::vector<int> texW, texH;
};

class Sphere: public Geometry {
public:

    Sphere(const Vector& center,
        double radius,
        const Vector& albedo = Vector(1, 1, 1),
        bool isMirror = false,
        bool isTransparent = false,
        double refraction_index = 1.0) :
        center(center), radius(radius), Geometry(albedo, isMirror, isTransparent, refraction_index) {};

    bool intersect(const Ray& ray, Vector& intersection_point, Vector& intersection_normal, double& t, Vector& albedo) const {

        double a{ 1 };
        double b = 2 * dot(ray.direction, ray.origin - center);
        double c = (ray.origin - center).norm2() - sqr(radius);

        double delta{ sqr(b) - 4 * a * c };

        if (delta < 0) return false; // pas d'intersection

        double sqrt_delta{ sqrt(delta) };
        double t1{ (-b - sqrt_delta) / (2 * a) };
        double t2{ (-b + sqrt_delta) / (2 * a) };

        if (t2 < 0) return false; // intersection pas du bon côté de la caméra

        if (t1 >= 0) {
            t = t1;
        } else {
            t = t2; // la caméra est dans la sphère
        }

        intersection_point = ray.origin + t * ray.direction; // point d'intersection
        intersection_normal = intersection_point - center; // vecteur normal à la sphère en intersection_point
        intersection_normal.normalize();
        
        albedo = this->albedo;

        // Damier sur les sphères:
        
        /*
        int large_offset{ 100000 }; // pour enlever les rangées de carreaux "doubles" en aux niveaux des axes, idée de Gulliver Larsonneur
        int frequency{ 5 };
        int alb0{ (int)(fmod(abs(intersection_point[0] + large_offset), 2 * frequency) < frequency) },
            alb1{ (int)(fmod(abs(intersection_point[1] + large_offset), 2 * frequency) < frequency) },
            alb2{ (int)(fmod(abs(intersection_point[2] + large_offset), 2 * frequency) < frequency) };

        albedo = albedo * ((alb0 + alb1 + alb2) % 2);
        */

        return true;
    }

    Vector center;
    double radius;
};

class LightSource {
public:

    LightSource(const Vector& position, const Vector& relative_intensity = Vector(1,1,1)): position(position), intensity(MAX_LIGHT_INTENSITY * relative_intensity) {};

    Vector position, intensity;
};

class Scene {
public:

    ~Scene() {
        // for (int i{ 0 }; i < objects.size(); i++) delete objects[i]; // segfault : le mesh est déjà delete
    }

    void addSphere(const Sphere * sphere) {
        objects.push_back(sphere);
    };

    void addMesh(const TriangleMesh * mesh) {
        objects.push_back(mesh);
    };

    Vector getColor(const Ray& ray, int nb_rebound = DEFAULT_MAX_RECURSION_DEPTH, bool is_indirect = false) {

        if (nb_rebound <= 0) return Vector(0, 0, 0); // trop de reflexions => on renvoie du noir

        int first_intersection_index{ 0 };
        double smallest_t{ std::numeric_limits<double>::max() };
        double t{ std::numeric_limits<double>::max() };
        bool intersected_once{ false };
        Vector intersection_point, intersection_normal;
        Vector object_color;

        for (int i{ 0 }; i < objects.size(); i++) {

            bool intersected{ objects[i]->intersect(ray, intersection_point, intersection_normal, t, object_color) };

            if (t < smallest_t) {
                smallest_t = t;
                first_intersection_index = i; // "premier" <=> le plus petit t (1er objet rencontré par le rayon)
            }

            if (intersected) intersected_once = true;
        }

        if (intersected_once) {

            Vector intersection_point_eps;
            const Geometry * intersected_object = objects[first_intersection_index];

            if (first_intersection_index == 0) {

                const Sphere* light_sphere{ dynamic_cast<const Sphere*>(intersected_object) };

                if (is_indirect) return Vector(0, 0, 0);

                return MAX_LIGHT_INTENSITY * intersected_object->albedo / (4 * sqr(M_PI) * sqr(light_sphere->radius));
            }
                
            intersected_object->intersect(ray, intersection_point, intersection_normal, t, object_color);

            bool total_reflection{ false };
            double dot_prod{ dot(ray.direction, intersection_normal) };

            if (dot_prod < 0) intersection_point_eps = intersection_point + EPSILON * intersection_normal;
            else intersection_point_eps = intersection_point - EPSILON * intersection_normal;

            if (intersected_object->isTransparent) {

                double refraction_index_ratio;
                Vector new_direction_tangential;
                double normal_comp_squared;
                double sign_normal;
                Vector new_direction_normal;
                Vector new_direction;
                Vector intersection_point_eps_t;

                double k0{ (refraction_index_void - intersected_object->refraction_index) / (refraction_index_void + intersected_object->refraction_index) };
                k0 = k0 * k0;
                double R{ k0 + (1 - k0) * pow(1 - abs(dot_prod), 5) };
                // double T{ 1 - R };

                if (dot_prod < 0) { // le rayon entre dans la sphère

                    refraction_index_ratio = refraction_index_void / intersected_object->refraction_index;
                    normal_comp_squared = 1 - refraction_index_ratio * refraction_index_ratio * (1 - dot_prod * dot_prod);
                    sign_normal = -1;
                    intersection_point_eps_t = intersection_point - EPSILON * intersection_normal;
                }
                else { // le rayon sort dans la sphère

                    refraction_index_ratio = intersected_object->refraction_index / refraction_index_void;
                    normal_comp_squared = 1 - refraction_index_ratio * refraction_index_ratio * (1 - dot_prod * dot_prod);
                    sign_normal = 1;
                    intersection_point_eps_t = intersection_point + EPSILON * intersection_normal;
                }

                if (normal_comp_squared < 0) total_reflection = true;
                else {
                    #ifdef _OPENMP
                        int thread_id{ omp_get_thread_num() };
                    #else
                        int thread_id{ 0 };
                    #endif

                    if (uniform(engines[thread_id]) < R) {
                        // Réflexion
                        Vector reflection_direction = ray.direction - 2 * dot_prod * intersection_normal;
                        Ray mirror_ray(intersection_point_eps, reflection_direction);
                        return getColor(mirror_ray, nb_rebound - 1);
                    }
                    else {
                        // Réfraction 
                        new_direction_tangential = refraction_index_ratio * (ray.direction - dot_prod * intersection_normal);
                        new_direction_normal = sign_normal * sqrt(normal_comp_squared) * intersection_normal;
                        new_direction = new_direction_normal + new_direction_tangential;
                        Ray refracted_ray(intersection_point_eps_t, new_direction);
                        return getColor(refracted_ray, nb_rebound - 1);
                    }
                }
            }

            if (intersected_object->isMirror || total_reflection) {

                Vector reflection_direction{ ray.direction - 2 * dot_prod * intersection_normal };
                Ray mirror_ray(intersection_point_eps, reflection_direction);

                return getColor(mirror_ray, nb_rebound - 1);
            }

            // Cas d'un objet diffus (par défaut) : 

            Vector color_direct(0, 0, 0);

            // Choix d'une point X considéré comme source de lumière sur la sphère
            const Sphere * light_sphere{ dynamic_cast<const Sphere *>(objects[0]) };
            Vector intersection_point_to_light_center{ light_sphere->center - intersection_point }; // PL
            intersection_point_to_light_center.normalize();
            Vector light_normal{ random_cos(-1 * intersection_point_to_light_center) };
            light_normal.normalize();
            Vector light_source{ light_sphere->center + light_normal * light_sphere->radius }; // X
            Vector light_source_eps{light_source + EPSILON * light_normal};


            // Lancer de rayon pour déterminer si le point d'intersection est à l'ombre de la source de lumière ou non
            Vector shadow_direction{ light_source_eps - intersection_point_eps };
            double shadow_dist{ sqrt(shadow_direction.norm2()) };
            shadow_direction.normalize();

            double light_visibility{ 1 };
            int first_shadow_intersection_index{ 0 };
            Ray shadow_ray(intersection_point_eps, shadow_direction);
            Vector shadow_intersection_point, shadow_intersection_normal;

            // Reset des variables réutilisables pour le 2ème lancer de rayon :
            smallest_t = std::numeric_limits<double>::max();
            t = std::numeric_limits<double>::max();
            intersected_once = false;

            for (int i{ 0 }; i < objects.size(); i++) {

                Vector dummy;
                bool shadow_ray_intersected{ objects[i]->intersect(shadow_ray, shadow_intersection_point, shadow_intersection_normal, t, dummy) };

                if (t < smallest_t) {
                    smallest_t = t;
                    first_shadow_intersection_index = i;
                }

                if (shadow_ray_intersected) intersected_once = true;
            }

            if (intersected_once && smallest_t <= shadow_dist) light_visibility = 0; // si intersection avant la source de lumière, pas de visibilité sur celle-ci

            Vector light_intensity{ MAX_LIGHT_INTENSITY * light_sphere->albedo }; // to mimic colored light
            color_direct = object_color * light_intensity / (4 * sqr(M_PI));
            color_direct = color_direct * dot(shadow_direction, intersection_normal) / dot(-1 * intersection_point_to_light_center, light_normal);
            color_direct = color_direct * dot(-1 * shadow_direction, light_normal) * light_visibility / sqr(shadow_dist);


            // Lumière indirecte
            Vector color_indirect(0, 0, 0);
            Vector random_direction{ random_cos(intersection_normal) };
            Ray random_ray(intersection_point_eps, random_direction);

            color_indirect = object_color * getColor(random_ray, nb_rebound - 1, true);

            return color_direct + color_indirect;
        }

        else return Vector(0, 0, 0); // couleur par défaut si pas d'intersection ciel ("sky") noir
    };

    double refraction_index_void{ 1.0 };
    std::vector<const Geometry*> objects;
};

int main() {

    int W{ WIDTH };
    int H{ HEIGHT };
    double alpha{ 60 * M_PI / 180 };
    double focus_distance{ 55 }; // 45
    double aperture_radius{ 0.1 };

    double angleUp = 0; // -30 * M_PI / 180;
    Vector cameraUp(0, cos(angleUp), sin(angleUp)), cameraDir(0, -sin(angleUp), cos(angleUp));
    Vector cameraRight{ cross(cameraUp, cameraDir) };

    int sphere_radius{ 10 };
    int offset_to_wall{ 50 };
    int big_radius{ 100000 };

    Vector origin_camera(0, 0, focus_distance); // 0, 25, ...
    Scene scene;

    scene.addSphere(new Sphere(Vector(15, 40, -35), 5, Vector(1, 1, 1))); // première sphère = la lumière
    
    TriangleMesh mesh;
    mesh.readOBJ("cat.obj");
    TriangleMesh * mesh_ptr = &mesh;
    scene.addMesh(mesh_ptr);
    mesh.transform(0.6, Vector(0, -10, 0));
    mesh.bbox = mesh.compute_bbox(0, mesh.indices.size());
    mesh.build_BVH(&mesh.root, 0, mesh.indices.size());

    mesh.add_texture("cat_diff.png");
    
    /*
    scene.addSphere(new Sphere(Vector(-5, 0, 0), sphere_radius, Vector(0.5, 0.2, 0.9)));
    //scene.addSphere(new Sphere(Vector(-20, 0, -15), sphere_radius, Vector(0.5, 0.9, 0.2), true));
    scene.addSphere(new Sphere(Vector(-20, 0, -15), sphere_radius, Vector(0.5, 0.9, 0.2)));
    //scene.addSphere(new Sphere(Vector(10, 0, 15), sphere_radius, Vector(0.9, 0.5, 0.2), false, true, 1.3));
    scene.addSphere(new Sphere(Vector(10, 0, 15), sphere_radius, Vector(0.9, 0.5, 0.2)));
    */

    scene.addSphere(new Sphere(Vector(big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(55.0 / 255.0, 215.0 / 255.0, 0.0 / 255.0)));
    //scene.addSphere(new Sphere(Vector(-big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(255.0 / 255.0, 140.0 / 255.0, 0.0 / 255.0), true));
    scene.addSphere(new Sphere(Vector(-big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(255.0 / 255.0, 140.0 / 255.0, 0.0 / 255.0)));
    scene.addSphere(new Sphere(Vector(0, big_radius, 0), big_radius - offset_to_wall - sphere_radius, Vector(238.0 / 255.0, 29.0 / 255.0, 35.0 / 255.0)));
    scene.addSphere(new Sphere(Vector(0, -big_radius, 0), big_radius - sphere_radius, Vector(0.0 / 255.0, 44.0 / 255.0, 89.0 / 255.0)));
    scene.addSphere(new Sphere(Vector(0, 0, -big_radius), big_radius - offset_to_wall - sphere_radius, Vector(56.0 / 255.0, 224.0 / 255.0, 116.0 / 255.0)));
    scene.addSphere(new Sphere(Vector(0, 0, big_radius), big_radius - offset_to_wall - sphere_radius, Vector(255 / 255.0, 255 / 255.0, 0 / 255.0)));

    std::vector<unsigned char> image(W*H * 3, 0);

    int counter{ 0 };

#ifdef _OPENMP
    for (int k{ 0 }; k < num_cores; k++) engines[k].seed(k);
    std::cout << "OpenMP is used. Parallelism on " << num_cores << " threads" << std::endl;
#else
    std::cout << "OpenMP is not used. Parallelism is disabled" << std::endl;
#endif

std::chrono::high_resolution_clock::time_point start{ std::chrono::high_resolution_clock::now() }; // début du chrono

#pragma omp parallel for num_threads(num_cores) schedule(dynamic, 1)
    for (int i{ 0 }; i < H; i++) {
        for (int j{ 0 }; j < W; j++) {
            
            counter += 1;
            if ((counter % (W * H / 10)) == 0) std::cout << 1 + (100 * counter) / (W * H) << "%" << std::endl;

            Vector color(0, 0, 0);

            #ifdef _OPENMP
                int thread_id{ omp_get_thread_num() };
            #else
                int thread_id{ 0 };
            #endif

            for (int k{ 0 }; k < NB_RAY; k++) {

                double dx, dy;
                boxMuller(dx, dy);
                Vector direction((j + 0.5 + dx) - (W / 2), (H / 2) - (i + 0.5 + dy), -W / (2 * tan(alpha / 2)));
                direction.normalize();

                #ifdef _OPENMP
                    int thread_id{ omp_get_thread_num() };
                #else
                    int thread_id{ 0 };
                #endif

                double dr_aperture{ aperture_radius * uniform(engines[thread_id]) };
                double dtheta_aperture{ 2 * M_PI * uniform(engines[thread_id]) };
                double dx_aperture{ dr_aperture * cos(dtheta_aperture) };
                double dy_aperture{ dr_aperture * sin(dtheta_aperture) };
                
                Vector destination{origin_camera + focus_distance * direction};
                Vector new_origin_camera{origin_camera + Vector(dx_aperture, dy_aperture, 0)};
                Vector new_direction{destination - new_origin_camera};
                new_direction.normalize();

                new_direction = new_direction[0] * cameraRight + new_direction[1] * cameraUp + new_direction[2] * cameraDir;

                Ray ray(new_origin_camera, new_direction);

                color += (scene.getColor(ray) / NB_RAY);
            }

            image[(i*W + j) * 3 + 0] = color_correction(color[0]); // RED
            image[(i*W + j) * 3 + 1] = color_correction(color[1]); // GREEN
            image[(i*W + j) * 3 + 2] = color_correction(color[2]); // BLUE
        }
    }

    std::chrono::high_resolution_clock::time_point stop{ std::chrono::high_resolution_clock::now() }; // fin du chrono
    std::chrono::seconds elapsed_time{ std::chrono::duration_cast<std::chrono::seconds>(stop - start) };
    std::cout << "Time taken: " << elapsed_time.count() << " seconds." << std::endl;

    stbi_write_png("image.png", W, H, 3, &image[0], 0); // on ne chronomètre pas le temps d'écriture en png
    return 0;
}
