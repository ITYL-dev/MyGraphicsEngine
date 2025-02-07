#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <limits>
#include <random>

#define WIDTH 512
#define HEIGHT 512
#define M_PI 3.14159265358
#define MAX_LIGHT_INTENSITY 1e10
#define GAMMA 2.2
#define EPSILON 1e-6
#define DEFAULT_MAX_RECURSION_DEPTH 8
#define NB_RAY 1028
#define DEFAULT_STD_ANTIALIASING 0.6

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
    else if (num < 0) return 0; // clamping clamping inférieur
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

class Sphere {
public:

    Sphere(
        const Vector& center,
        double radius,
        const Vector& albedo = Vector(1,1,1),
        bool isMirror = false,
        bool isTransparent = false,
        double refraction_index = 1.0) :
        center(center), radius(radius), albedo(albedo), isMirror(isMirror), isTransparent(isTransparent), refraction_index(refraction_index) { };

    bool intersect(const Ray& ray, Vector& intersection_point, Vector& intersection_normal, double& t) {
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

        return true;
    }

    Vector center;
    Vector albedo;
    double radius;
    bool isMirror;
    bool isTransparent;
    double refraction_index;
};

class LightSource {
public:

    LightSource(const Vector& position, const Vector& relative_intensity = Vector(1,1,1)): position(position), intensity(MAX_LIGHT_INTENSITY * relative_intensity) {};

    Vector position, intensity;
};

class Scene {
public:

    void addSphere(const Sphere& sphere) {
        objects.push_back(sphere);
    };

    void addLight(const LightSource& light_source) {
        light_sources.push_back(light_source);
    };

    Vector getColor(const Ray& ray, int nb_rebound = DEFAULT_MAX_RECURSION_DEPTH, bool is_indirect = false) {

        if (nb_rebound <= 0) return Vector(0, 0, 0); // trop de reflexions => on renvoie du noir

        int first_intersection_index{ 0 };
        double smallest_t{ std::numeric_limits<double>::max() };
        double t{ std::numeric_limits<double>::max() };
        bool intersected_once{ false };
        Vector intersection_point, intersection_normal;

        for (int i{ 0 }; i < objects.size(); i++) {

            bool intersected{ objects[i].intersect(ray, intersection_point, intersection_normal, t) };

            if (t < smallest_t) {
                smallest_t = t;
                first_intersection_index = i; // "premier" <=> le plus petit t (1er objet rencontré par le rayon)
            }

            if (intersected) intersected_once = true;
        }

        if (intersected_once) {

            Vector intersection_point_eps;
            Sphere intersected_sphere{ objects[first_intersection_index] };

            if (first_intersection_index == 0) {

                if (is_indirect) return Vector(0, 0, 0);

                return MAX_LIGHT_INTENSITY * intersected_sphere.albedo / (4 * sqr(M_PI) * sqr(intersected_sphere.radius));
            }
                

            intersected_sphere.intersect(ray, intersection_point, intersection_normal, t);

            bool total_reflection{ false };
            double dot_prod{ dot(ray.direction, intersection_normal) };

            if (dot_prod < 0) intersection_point_eps = intersection_point + EPSILON * intersection_normal;
            else intersection_point_eps = intersection_point - EPSILON * intersection_normal;

            if (intersected_sphere.isTransparent) {

                double refraction_index_ratio;
                Vector new_direction_tangential;
                double normal_comp_squared;
                double sign_normal;
                Vector new_direction_normal;
                Vector new_direction;
                Vector intersection_point_eps_t;

                double k0{ (refraction_index_void - intersected_sphere.refraction_index) / (refraction_index_void + intersected_sphere.refraction_index) };
                k0 = k0 * k0;
                double R{ k0 + (1 - k0) * pow(1 - abs(dot_prod), 5) };
                // double T{ 1 - R };

                if (dot_prod < 0) { // le rayon entre dans la sphère

                    refraction_index_ratio = refraction_index_void / intersected_sphere.refraction_index;
                    normal_comp_squared = 1 - refraction_index_ratio * refraction_index_ratio * (1 - dot_prod * dot_prod);
                    sign_normal = -1;
                    intersection_point_eps_t = intersection_point - EPSILON * intersection_normal;
                }
                else { // le rayon sort dans la sphère

                    refraction_index_ratio = intersected_sphere.refraction_index / refraction_index_void;
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

            if (intersected_sphere.isMirror || total_reflection) {

                Vector reflection_direction{ ray.direction - 2 * dot_prod * intersection_normal };
                Ray mirror_ray(intersection_point_eps, reflection_direction);

                return getColor(mirror_ray, nb_rebound - 1);
            }

            // Cas d'un objet diffus (par défaut) : 

            Vector color_direct(0, 0, 0);

            /*

            for (int l{ 0 }; l < light_sources.size(); l++) {

                // Lancer de rayon pour déterminer si le point d'intersection est à l'ombre de la source de lumière ou non
                double light_visibility{ 1 };
                int first_shadow_intersection_index{ 0 };
                Vector shadow_direction{ light_sources[l].position - intersection_point_eps };
                shadow_direction.normalize();
                Ray shadow_ray(intersection_point_eps, shadow_direction);
                Vector shadow_intersection_point, shadow_intersection_normal;

                // Reset des variables réutilisables pour le 2ème lancer de rayon :
                smallest_t = std::numeric_limits<double>::max();
                t = std::numeric_limits<double>::max();
                intersected_once = false;

                for (int i{ 0 }; i < objects.size(); i++) {

                    bool shadow_ray_intersected{ objects[i].intersect(shadow_ray, shadow_intersection_point, shadow_intersection_normal, t) };

                    if (t < smallest_t) {
                        smallest_t = t;
                        first_shadow_intersection_index = i;
                    }

                    if (shadow_ray_intersected) intersected_once = true;
                }

                if (intersected_once && smallest_t <= sqrt((light_sources[l].position - intersection_point_eps).norm2())) light_visibility = 0; // si intersection avant la source de lumière, pas de visibilité sur celle-ci

                Vector normalized_dist{ light_sources[l].position - intersection_point_eps };
                normalized_dist.normalize();

                double common_factor = light_visibility * dot(intersection_normal, normalized_dist) / (4 * sqr(M_PI) * (light_sources[l].position - intersection_point_eps).norm2());
                color_direct += common_factor * light_sources[l].intensity * intersected_sphere.albedo;
            }
            */

            // Choix d'une point X considéré comme source de lumière sur la sphère
            Sphere light_sphere{ objects[0] };
            Vector intersection_point_to_light_center{ light_sphere.center - intersection_point }; // PL
            intersection_point_to_light_center.normalize();
            Vector light_normal{ random_cos(-1 * intersection_point_to_light_center) };
            light_normal.normalize();
            Vector light_source{ light_sphere.center + light_normal * light_sphere.radius }; // X
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

                bool shadow_ray_intersected{ objects[i].intersect(shadow_ray, shadow_intersection_point, shadow_intersection_normal, t) };

                if (t < smallest_t) {
                    smallest_t = t;
                    first_shadow_intersection_index = i;
                }

                if (shadow_ray_intersected) intersected_once = true;
            }

            if (intersected_once && smallest_t <= shadow_dist) light_visibility = 0; // si intersection avant la source de lumière, pas de visibilité sur celle-ci

            Vector light_intensity{ MAX_LIGHT_INTENSITY * light_sphere.albedo }; // to mimic colored light
            color_direct = intersected_sphere.albedo * light_intensity / (4 * sqr(M_PI));
            color_direct = color_direct * dot(shadow_direction, intersection_normal) / dot(-1 * intersection_point_to_light_center, light_normal);
            color_direct = color_direct * dot(-1 * shadow_direction, light_normal) * light_visibility / sqr(shadow_dist);


            // Lumière indirecte
            Vector color_indirect(0, 0, 0);
            Vector random_direction{ random_cos(intersection_normal) };
            Ray random_ray(intersection_point_eps, random_direction);

            color_indirect = intersected_sphere.albedo * getColor(random_ray, nb_rebound - 1, true);

            return color_direct + color_indirect;
        }

        else return Vector(0, 0, 0); // couleur par défaut si pas d'intersection ciel ("sky") noir
    };

    double refraction_index_void{ 1.0 };
    std::vector<Sphere> objects;
    std::vector<LightSource> light_sources;
};

int main() {

    int W{ WIDTH };
    int H{ HEIGHT };
    double alpha{ 60 * M_PI / 180 };
    double focus_distance{ 55 };
    double aperture_radius{ 1.5 };

    int sphere_radius{ 10 };
    int offset_to_wall{ 50 };
    int big_radius{ 100000 };

    Vector origin_camera(0, 0, focus_distance);
    Scene scene;

    scene.addSphere(Sphere(Vector(15, 35, -35), 7.5, Vector(1, 1, 1)));
    
    scene.addSphere(Sphere(Vector(-5, 0, 0), sphere_radius, Vector(0.5, 0.2, 0.9)));
    //scene.addSphere(Sphere(Vector(-10, 0, -10), sphere_radius, Vector(0.5, 0.9, 0.2)));
    scene.addSphere(Sphere(Vector(-20, 0, -15), sphere_radius, Vector(0.5, 0.9, 0.2), true));
    //scene.addSphere(Sphere(Vector(10, 0, 10), sphere_radius, Vector(0.9, 0.5, 0.2)));
    scene.addSphere(Sphere(Vector(10, 0, 15), sphere_radius, Vector(0.9, 0.5, 0.2), false, true, 1.3));

    // Couleurs des murs utilisées par le prof
    /*scene.addSphere(Sphere(Vector(big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(0.8, 0.4, 0.6)));
    scene.addSphere(Sphere(Vector(-big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(0.2, 0.3, 0.8)));
    scene.addSphere(Sphere(Vector(0, big_radius, 0), big_radius - offset_to_wall - sphere_radius, Vector(0.6, 0.8, 0.7)));
    scene.addSphere(Sphere(Vector(0, -big_radius, 0), big_radius - sphere_radius, Vector(0.4, 0.8, 0.5)));
    scene.addSphere(Sphere(Vector(0, 0, -big_radius), big_radius - offset_to_wall - sphere_radius, Vector(0.4, 0.4, 0.9)));
    scene.addSphere(Sphere(Vector(0, 0, big_radius), big_radius - offset_to_wall - sphere_radius, Vector(0.9, 0.8, 0.5)));*/

    scene.addSphere(Sphere(Vector(big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(55.0 / 255.0, 215.0 / 255.0, 0.0 / 255.0)));
    scene.addSphere(Sphere(Vector(-big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(255.0 / 255.0, 140.0 / 255.0, 0.0 / 255.0)));
    scene.addSphere(Sphere(Vector(0, big_radius, 0), big_radius - offset_to_wall - sphere_radius, Vector(238.0 / 255.0, 29.0 / 255.0, 35.0 / 255.0)));
    scene.addSphere(Sphere(Vector(0, -big_radius, 0), big_radius - sphere_radius, Vector(0.0 / 255.0, 44.0 / 255.0, 89.0 / 255.0)));
    scene.addSphere(Sphere(Vector(0, 0, -big_radius), big_radius - offset_to_wall - sphere_radius, Vector(56.0 / 255.0, 224.0 / 255.0, 116.0 / 255.0)));
    scene.addSphere(Sphere(Vector(0, 0, big_radius), big_radius - offset_to_wall - sphere_radius, Vector(255 / 255.0, 255 / 255.0, 0 / 255.0)));

    std::vector<unsigned char> image(W*H * 3, 0);

    int counter{ 0 };

#ifdef _OPENMP
    for (int k{ 0 }; k < num_cores; k++) engines[k].seed(k);
    std::cout << "OpenMP is used. Parallelism on " << num_cores << " threads" << std::endl;
#else
    std::cout << "OpenMP is not used. Parallelism is disabled" << std::endl;
#endif

#pragma omp parallel for num_threads(num_cores)
    for (int i{ 0 }; i < H; i++) {
        for (int j{ 0 }; j < W; j++) {
            
            counter += 1;
            if ((counter % (W * H / 10)) == 0) std::cout << 1 + (100 * counter) / (W * H) << "%" << std::endl;

            Vector color(0, 0, 0);

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

                Ray ray(new_origin_camera, new_direction);

                color += (scene.getColor(ray) / NB_RAY);
            }

            image[(i*W + j) * 3 + 0] = color_correction(color[0]); // RED
            image[(i*W + j) * 3 + 1] = color_correction(color[1]); // GREEN
            image[(i*W + j) * 3 + 2] = color_correction(color[2]); // BLUE
        }
    }

    stbi_write_png("image.png", W, H, 3, &image[0], 0);
    return 0;
}
