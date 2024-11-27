import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

def generate_sphere_cluster(center: np.ndarray, num_points: int, spread: float = 0.2) -> np.ndarray:
    vectors = np.random.randn(num_points, 3)
    proj = vectors - np.dot(vectors, center.reshape(-1, 1)) * center
    points = center + spread * proj
    return points / np.linalg.norm(points, axis=1, keepdims=True)

def generate_dataset(num_clusters: int = 3, points_per_cluster: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.array([
        [1.0, 0.0, 0.0],
        [-0.5, 0.866, 0.0],
        [-0.5, -0.866, 0.0]
    ])
    
    points, labels = [], []
    for i in range(num_clusters):
        points.append(generate_sphere_cluster(centers[i], points_per_cluster))
        labels.extend([i] * points_per_cluster)
    
    return np.vstack(points), np.array(labels)

def plot_sphere_data(points: np.ndarray, labels: np.ndarray, weights: np.ndarray = None, title: str = "Data Points on Unit Sphere"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.1)
    
    colors = ['r', 'g', 'b']
    for i in range(3):
        cluster = points[labels == i]
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=colors[i], label=f'Cluster {i+1}')
    
    if weights is not None:
        for i, w in enumerate(weights):
            ax.plot([0, w[0]], [0, w[1]], [0, w[2]], color=colors[i], linewidth=2)
            ax.plot([w[0]], [w[1]], [w[2]], color=colors[i], marker='x', markersize=10, label=f'Weight {i+1}')
    
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def generate_random_weights(num_clusters: int = 3) -> np.ndarray:
    weights = np.random.randn(num_clusters, 3)
    return weights / np.linalg.norm(weights, axis=1, keepdims=True)

def find_winner(input_vector: np.ndarray, weights: np.ndarray) -> int:
    return np.argmax(np.dot(weights, input_vector))

def train_network(points: np.ndarray, labels: np.ndarray, learning_rate: float = 0.1, 
                 max_epochs: int = 100) -> Tuple[np.ndarray, List[np.ndarray]]:
    weights = generate_random_weights()
    weight_history = [weights.copy()]
    
    for epoch in range(max_epochs):
        points_shuffled = points[np.random.permutation(len(points))]
        weights_changed = False
        
        for input_vector in points_shuffled:
            winner = find_winner(input_vector, weights)
            
            delta = learning_rate * (input_vector - weights[winner])
            weights[winner] += delta
            weights[winner] /= np.linalg.norm(weights[winner])
            weights_changed = weights_changed or np.linalg.norm(delta) > 1e-7
            weight_history.append(weights.copy())
        
        if not weights_changed:
            break
    
    return weights, weight_history

def plot_weight_trajectories(weight_history: List[np.ndarray], points: np.ndarray, labels: np.ndarray):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)), 
                   np.outer(np.sin(u), np.sin(v)), 
                   np.outer(np.ones(np.size(u)), np.cos(v)), 
                   color='gray', alpha=0.1)
    
    cluster_colors = ['r', 'g', 'b']
    weight_colors = ['purple', 'orange', 'cyan']
    
    for i in range(3):
        cluster = points[labels == i]
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=cluster_colors[i], alpha=0.3, label=f'Cluster {i+1}')
        
        trajectory = np.array(weight_history)[:, i, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=weight_colors[i], alpha=0.5, label=f'Weight {i+1} trajectory')
    
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_title('Weight Vector Trajectories During Training')
    ax.legend()
    plt.show()

def test_model(weights: np.ndarray, test_points: np.ndarray, test_labels: np.ndarray):
    predictions = np.array([find_winner(point, weights) for point in test_points])
    
    # Find the mapping between weight indices and original cluster labels
    cluster_centers = np.array([
        [1.0, 0.0, 0.0],
        [-0.5, 0.866, 0.0],
        [-0.5, -0.866, 0.0]
    ])
    
    weight_to_cluster = np.array([np.argmax(np.dot(w, cluster_centers.T)) for w in weights])
    remapped_predictions = np.array([weight_to_cluster[p] for p in predictions])
    
    accuracy = np.mean(remapped_predictions == test_labels) * 100
    print(f"Classification accuracy: {accuracy:.2f}%")
    for i, point in enumerate(test_points):
        outputs = np.dot(weights, point)
        print(f"Sample {i+1}: Outputs=[{outputs[0]:.3f}, {outputs[1]:.3f}, {outputs[2]:.3f}] Winner=Cluster {remapped_predictions[i]+1}")
    
    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)), 
                   np.outer(np.sin(u), np.sin(v)), 
                   np.outer(np.ones(np.size(u)), np.cos(v)), 
                   color='gray', alpha=0.1)
    
    cluster_colors = ['r', 'g', 'b']
    weight_colors = ['purple', 'orange', 'cyan']
    
    # Plot test points colored by their predicted cluster
    for i in range(3):
        cluster_points = test_points[remapped_predictions == i]
        cluster_indices = np.where(remapped_predictions == i)[0]
        if len(cluster_points) > 0:
            scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                               c=cluster_colors[i], s=100, label=f'Cluster {i+1}')
            # Add text labels for each point
            for point, idx in zip(cluster_points, cluster_indices):
                ax.text(point[0], point[1], point[2], f' {idx+1}', fontsize=10)
    
    # Plot weight vectors with colors matching their closest clusters
    for i, w in enumerate(weights):
        cluster_idx = weight_to_cluster[i]
        ax.plot([0, w[0]], [0, w[1]], [0, w[2]], color=weight_colors[i], linewidth=2)
        ax.plot([w[0]], [w[1]], [w[2]], color=weight_colors[i], marker='x', markersize=10, 
                label=f'Weight {i+1}')
    
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_title('Test Results')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    points, labels = generate_dataset()
    
    test_indices = []
    for i in range(3):
        cluster_indices = np.where(labels == i)[0]
        test_indices.extend(np.random.choice(cluster_indices, size=3, replace=False))
    
    test_indices = np.array(test_indices)
    train_indices = np.array([i for i in range(len(points)) if i not in test_indices])
    
    train_points, train_labels = points[train_indices], labels[train_indices]
    test_points, test_labels = points[test_indices], labels[test_indices]
    
    final_weights, weight_history = train_network(train_points, train_labels)
    plot_weight_trajectories(weight_history, train_points, train_labels)
    test_model(final_weights, test_points, test_labels)