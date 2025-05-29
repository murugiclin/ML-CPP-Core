#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

// Linear Regression
class LinearRegression {
private:
    double slope;
    double intercept;

public:
    LinearRegression() : slope(0.0), intercept(0.0) {}

    void fit(const std::vector<double>& X, const std::vector<double>& y) {
        // Calculate means
        double x_mean = std::accumulate(X.begin(), X.end(), 0.0) / X.size();
        double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

        // Calculate slope
        double numerator = 0.0, denominator = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            numerator += (X[i] - x_mean) * (y[i] - y_mean);
            denominator += (X[i] - x_mean) * (X[i] - x_mean);
        }
        slope = numerator / denominator;
        intercept = y_mean - slope * x_mean;
    }

    double predict(double x) const {
        return slope * x + intercept;
    }

    void print_model() const {
        std::cout << "Linear Regression: y = " << slope << "x + " << intercept << "\n";
    }
};

// Decision Tree (for binary classification)
struct DecisionTreeNode {
    int feature_index = -1;
    double threshold = 0.0;
    int label = -1;
    DecisionTreeNode* left = nullptr;
    DecisionTreeNode* right = nullptr;
};

class DecisionTree {
private:
    DecisionTreeNode* root;
    int max_depth;

    DecisionTreeNode* build_tree(const std::vector<std::vector<double>>& X,
                               const std::vector<int>& y,
                               int depth) {
        if (depth >= max_depth || X.size() <= 1) {
            DecisionTreeNode* leaf = new DecisionTreeNode;
            leaf->label = most_common_label(y);
            return leaf;
        }

        double best_gain = -1.0;
        int best_feature = -1;
        double best_threshold = 0.0;

        // Simple feature selection
        for (size_t f = 0; f < X[0].size(); ++f) {
            std::vector<double> values;
            for (const auto& x : X) values.push_back(x[f]);
            std::sort(values.begin(), values.end());
            double threshold = values[values.size() / 2];

            double gain = calculate_gini_gain(X, y, f, threshold);
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = threshold;
            }
        }

        if (best_gain <= 0) {
            DecisionTreeNode* leaf = new DecisionTreeNode;
            leaf->label = most_common_label(y);
            return leaf;
        }

        DecisionTreeNode* node = new DecisionTreeNode;
        node->feature_index = best_feature;
        node->threshold = best_threshold;

        std::vector<std::vector<double>> left_X, right_X;
        std::vector<int> left_y, right_y;
        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][best_feature] <= best_threshold) {
                left_X.push_back(X[i]);
                left_y.push_back(y[i]);
            } else {
                right_X.push_back(X[i]);
                right_y.push_back(y[i]);
            }
        }

        node->left = build_tree(left_X, left_y, depth + 1);
        node->right = build_tree(right_X, right_y, depth + 1);
        return node;
    }

    double calculate_gini_gain(const std::vector<std::vector<double>>& X,
                             const std::vector<int>& y,
                             int feature,
                             double threshold) {
        std::vector<int> left, right;
        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][feature] <= threshold) left.push_back(y[i]);
            else right.push_back(y[i]);
        }

        double parent_gini = calculate_gini(y);
        double left_gini = calculate_gini(left);
        double right_gini = calculate_gini(right);
        double weighted_gini = (left.size() * left_gini + right.size() * right_gini) / y.size();
        return parent_gini - weighted_gini;
    }

    double calculate_gini(const std::vector<int>& labels) {
        if (labels.empty()) return 0.0;
        int count_0 = std::count(labels.begin(), labels.end(), 0);
        double p0 = static_cast<double>(count_0) / labels.size();
        double p1 = 1.0 - p0;
        return 1.0 - (p0 * p0 + p1 * p1);
    }

    int most_common_label(const std::vector<int>& labels) {
        if (labels.empty()) return 0;
        int count_0 = std::count(labels.begin(), labels.end(), 0);
        return count_0 > labels.size() / 2 ? 0 : 1;
    }

public:
    DecisionTree(int depth = 3) : max_depth(depth), root(nullptr) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        root = build_tree(X, y, 0);
    }

    int predict(const std::vector<double>& x) const {
        DecisionTreeNode* current = root;
        while (current->label == -1) {
            if (x[current->feature_index] <= current->threshold)
                current = current->left;
            else
                current = current->right;
        }
        return current->label;
    }

    ~DecisionTree() {
        delete_tree(root);
    }

private:
    void delete_tree(DecisionTreeNode* node) {
        if (!node) return;
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
};

// K-Means Clustering
class KMeans {
private:
    int k;
    std::vector<std::vector<double>> centroids;

    double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

public:
    KMeans(int clusters) : k(clusters) {}

    void fit(const std::vector<std::vector<double>>& X, int max_iterations = 100) {
        // Initialize centroids randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, X.size() - 1);
        for (int i = 0; i < k; ++i) {
            centroids.push_back(X[dis(gen)]);
        }

        for (int iter = 0; iter < max_iterations; ++iter) {
            // Assign points to clusters
            std::vector<std::vector<int>> clusters(k);
            for (size_t i = 0; i < X.size(); ++i) {
                int closest = 0;
                double min_dist = euclidean_distance(X[i], centroids[0]);
                for (int j = 1; j < k; ++j) {
                    double dist = euclidean_distance(X[i], centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest = j;
                    }
                }
                clusters[closest].push_back(i);
            }

            // Update centroids
            std::vector<std::vector<double>> new_centroids(k, std::vector<double>(X[0].size(), 0.0));
            for (int j = 0; j < k; ++j) {
                if (clusters[j].empty()) continue;
                for (int idx : clusters[j]) {
                    for (size_t d = 0; d < X[0].size(); ++d) {
                        new_centroids[j][d] += X[idx][d];
                    }
                }
                for (size_t d = 0; d < X[0].size(); ++d) {
                    new_centroids[j][d] /= clusters[j].size();
                }
            }
            centroids = new_centroids;
        }
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& X) {
        std::vector<int> labels(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            int closest = 0;
            double min_dist = euclidean_distance(X[i], centroids[0]);
            for (int j = 1; j < k; ++j) {
                double dist = euclidean_distance(X[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = j;
                }
            }
            labels[i] = closest;
        }
        return labels;
    }
};

// Example usage
int main() {
    // Linear Regression Example
    std::vector<double> X = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    LinearRegression lr;
    lr.fit(X, y);
    lr.print_model();
    std::cout << "Prediction for x=6: " << lr.predict(6) << "\n";

    // Decision Tree Example
    std::vector<std::vector<double>> X_dt = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
    std::vector<int> y_dt = {0, 1, 0, 1};
    DecisionTree dt(3);
    dt.fit(X_dt, y_dt);
    std::cout << "Decision Tree prediction for [0.5, 0.5]: " << dt.predict({0.5, 0.5}) << "\n";

    // K-Means Example
    std::vector<std::vector<double>> X_km = {{1, 1}, {1.5, 2}, {3, 4}, {5, 7}, {3.5, 5}};
    KMeans km(2);
    km.fit(X_km);
    auto labels = km.predict(X_km);
    std::cout << "K-Means cluster assignments: ";
    for (int label : labels) std::cout << label << " ";
    std::cout << "\n";

    return 0;
}
