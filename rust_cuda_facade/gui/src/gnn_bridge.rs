/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

use crate::gnn_facade::CudaGnnFacade;
use cxx_qt::CxxQtType;

#[cxx_qt::bridge]
pub mod qobject {
    unsafe extern "C++" {
        include!("cxx-qt-lib/qstring.h");
        type QString = cxx_qt_lib::QString;
    }

    extern "RustQt" {
        #[qobject]
        #[qml_element]
        #[qproperty(i32, feature_size)]
        #[qproperty(i32, hidden_size)]
        #[qproperty(i32, output_size)]
        #[qproperty(i32, num_mp_layers)]
        #[qproperty(f64, learning_rate)]
        #[qproperty(QString, activation)]
        #[qproperty(QString, loss_function)]
        #[qproperty(i32, num_nodes)]
        #[qproperty(i32, num_edges)]
        #[qproperty(bool, network_created)]
        #[qproperty(bool, graph_loaded)]
        #[qproperty(QString, status_message)]
        #[qproperty(QString, facade_output)]
        #[qproperty(i32, facade_node_idx)]
        #[qproperty(i32, facade_edge_idx)]
        #[qproperty(i32, facade_layer_idx)]
        #[qproperty(i32, facade_feature_idx)]
        #[qproperty(i32, facade_neighbor_idx)]
        #[qproperty(i32, facade_neuron_idx)]
        #[qproperty(i32, facade_weight_idx)]
        #[qproperty(f64, facade_set_value)]
        #[qproperty(QString, edge_list)]
        #[qproperty(QString, target_output)]
        #[qproperty(i32, train_iterations)]
        #[qproperty(bool, undirected)]
        #[qproperty(bool, self_loops)]
        #[qproperty(QString, predict_output)]
        #[qproperty(f64, current_loss)]
        #[qproperty(i32, current_iteration)]
        #[qproperty(i32, total_iterations)]
        type GnnBridge = super::GnnBridgeRust;
    }

    unsafe extern "RustQt" {
        #[qinvokable]
        fn create_network(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn build_graph(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn randomize_features(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn train_network(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn predict_only(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn save_model(self: Pin<&mut GnnBridge>, filename: QString);

        #[qinvokable]
        fn load_model(self: Pin<&mut GnnBridge>, filename: QString);

        #[qinvokable]
        fn get_node_feature(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_node_features(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_edge_features(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_num_nodes_facade(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_num_edges_facade(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_neighbors(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_adjacency_matrix(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_edge_endpoints(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn has_edge(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_in_degree(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_out_degree(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_graph_embedding(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_node_mask(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn toggle_node_mask(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn apply_node_dropout(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn apply_edge_dropout(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_masked_counts(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn add_node(self: Pin<&mut GnnBridge>, features: QString);

        #[qinvokable]
        fn remove_node(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn add_edge_facade(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn remove_edge(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn clear_all_edges(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn rebuild_adjacency(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_node_degree(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn compute_page_rank(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_parameter_count(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_architecture_summary(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn export_graph_json(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn set_node_feature(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn set_learning_rate_facade(self: Pin<&mut GnnBridge>);

        #[qinvokable]
        fn get_graph_data(self: Pin<&mut GnnBridge>) -> QString;

        #[qinvokable]
        fn import_from_csv(self: Pin<&mut GnnBridge>, nodes_csv: QString, edges_csv: QString);

        #[qinvokable]
        fn apply_manual_features(self: Pin<&mut GnnBridge>, features_text: QString);
    }
}

use cxx_qt_lib::QString;
use std::pin::Pin;

pub struct GnnBridgeRust {
    feature_size: i32,
    hidden_size: i32,
    output_size: i32,
    num_mp_layers: i32,
    learning_rate: f64,
    activation: QString,
    loss_function: QString,
    num_nodes: i32,
    num_edges: i32,
    network_created: bool,
    graph_loaded: bool,
    status_message: QString,
    facade_output: QString,
    facade_node_idx: i32,
    facade_edge_idx: i32,
    facade_layer_idx: i32,
    facade_feature_idx: i32,
    facade_neighbor_idx: i32,
    facade_neuron_idx: i32,
    facade_weight_idx: i32,
    facade_set_value: f64,
    edge_list: QString,
    target_output: QString,
    train_iterations: i32,
    undirected: bool,
    self_loops: bool,
    predict_output: QString,
    current_loss: f64,
    current_iteration: i32,
    total_iterations: i32,

    facade: Option<CudaGnnFacade>,
    training_target: Vec<f32>,
}

impl Default for GnnBridgeRust {
    fn default() -> Self {
        Self {
            feature_size: 3,
            hidden_size: 16,
            output_size: 2,
            num_mp_layers: 2,
            learning_rate: 0.01,
            activation: QString::from("relu"),
            loss_function: QString::from("mse"),
            num_nodes: 5,
            num_edges: 0,
            network_created: false,
            graph_loaded: false,
            status_message: QString::from(""),
            facade_output: QString::from("Facade output will appear here..."),
            facade_node_idx: 0,
            facade_edge_idx: 0,
            facade_layer_idx: 0,
            facade_feature_idx: 0,
            facade_neighbor_idx: 0,
            facade_neuron_idx: 0,
            facade_weight_idx: 0,
            facade_set_value: 0.0,
            edge_list: QString::from("0,1\n1,2\n2,3\n3,4\n4,0\n1,3"),
            target_output: QString::from("1,0"),
            train_iterations: 200,
            undirected: true,
            self_loops: false,
            predict_output: QString::from(""),
            current_loss: 0.0,
            current_iteration: 0,
            total_iterations: 0,
            facade: None,
            training_target: Vec::new(),
        }
    }
}

impl qobject::GnnBridge {
    fn create_network(mut self: Pin<&mut Self>) {
        let feature = *self.as_ref().feature_size() as usize;
        let hidden = *self.as_ref().hidden_size() as usize;
        let output = *self.as_ref().output_size() as usize;
        let layers = *self.as_ref().num_mp_layers() as usize;
        let lr = *self.as_ref().learning_rate() as f32;

        match CudaGnnFacade::new(feature, hidden, output, layers) {
            Ok(mut facade) => {
                facade.set_learning_rate(lr);
                let param_count = facade.get_parameter_count();
                self.as_mut().rust_mut().facade = Some(facade);
                self.as_mut().set_network_created(true);
                self.as_mut().set_status_message(QString::from(&format!(
                    "Network created! {} MP layers, {} parameters",
                    layers, param_count
                )));
            }
            Err(e) => {
                self.as_mut().set_status_message(QString::from(&format!("Error creating network: {}", e)));
            }
        }
    }

    fn build_graph(mut self: Pin<&mut Self>) {
        if self.as_ref().rust().facade.is_none() {
            self.as_mut().set_status_message(QString::from("Create network first"));
            return;
        }

        let num_nodes = *self.as_ref().num_nodes() as usize;
        let feature_size = *self.as_ref().feature_size() as usize;
        let edge_text = self.as_ref().edge_list().to_string();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
                facade.create_empty_graph(num_nodes, feature_size);

                for line in edge_text.lines() {
                    let parts: Vec<&str> = line.trim().split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(s), Ok(t)) = (parts[0].trim().parse::<usize>(), parts[1].trim().parse::<usize>()) {
                            if s < num_nodes && t < num_nodes {
                                facade.add_edge(s, t, vec![]);
                            }
                        }
                    }
                }

                let mut rng = rand::thread_rng();
                use rand::Rng;
                for i in 0..num_nodes {
                    let features: Vec<f32> = (0..feature_size).map(|_| rng.gen()).collect();
                    facade.set_node_features(i, features);
                }

                facade.rebuild_adjacency_list();
                facade.get_num_edges()
            } else {
                0
            }
        }));

        match result {
            Ok(num_edges) => {
                self.as_mut().set_num_edges(num_edges as i32);
                self.as_mut().set_graph_loaded(true);
                self.as_mut().set_status_message(QString::from(&format!(
                    "Graph built: {} nodes, {} edges",
                    num_nodes, num_edges
                )));
            }
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                self.as_mut().set_status_message(QString::from(&format!("Error: {}", msg)));
            }
        }
    }

    fn randomize_features(mut self: Pin<&mut Self>) {
        let (num_nodes, feature_size) = {
            if let Some(facade) = self.as_ref().rust().facade.as_ref() {
                (facade.get_num_nodes(), facade.get_feature_size())
            } else {
                return;
            }
        };

        if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            let mut rng = rand::thread_rng();
            use rand::Rng;
            for i in 0..num_nodes {
                let features: Vec<f32> = (0..feature_size).map(|_| rng.gen()).collect();
                facade.set_node_features(i, features);
            }
        }

        self.as_mut().set_facade_output(QString::from("Features randomized"));
    }

    fn train_network(mut self: Pin<&mut Self>) {
        let graph_loaded = *self.as_ref().graph_loaded();
        if self.as_ref().rust().facade.is_none() || !graph_loaded {
            self.as_mut().set_status_message(QString::from("Build graph first"));
            return;
        }

        let target_text = self.as_ref().target_output().to_string();
        let target: Vec<f32> = target_text
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let iterations = *self.as_ref().train_iterations() as usize;

        let mut final_loss = 0.0f32;
        if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            for _ in 0..iterations {
                match facade.train(&target) {
                    Ok(l) => final_loss = l,
                    Err(e) => {
                        self.as_mut().set_status_message(QString::from(&format!("Training error: {}", e)));
                        return;
                    }
                }
            }
        }

        self.as_mut().set_current_loss(final_loss as f64);
        self.as_mut().set_status_message(QString::from(&format!(
            "Training complete. Final loss: {:.6}",
            final_loss
        )));
    }

    fn predict_only(mut self: Pin<&mut Self>) {
        let graph_loaded = *self.as_ref().graph_loaded();
        if self.as_ref().rust().facade.is_none() || !graph_loaded {
            self.as_mut().set_status_message(QString::from("Build graph first"));
            return;
        }

        let target_text = self.as_ref().target_output().to_string();
        let target: Vec<f32> = target_text
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let result = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.predict()
        } else {
            return;
        };

        match result {
            Ok(prediction) => {
                let pred_str: String = prediction
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
                    .join(", ");
                let target_str: String = target
                    .iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join(", ");

                self.as_mut().set_predict_output(QString::from(&format!(
                    "Prediction: [{}]\nTarget: [{}]",
                    pred_str, target_str
                )));
            }
            Err(e) => {
                self.as_mut().set_predict_output(QString::from(&format!("Prediction error: {}", e)));
            }
        }
    }

    fn save_model(mut self: Pin<&mut Self>, filename: QString) {
        let path = filename.to_string();
        let result = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.save_model(&path)
        } else {
            return;
        };

        match result {
            Ok(_) => {
                self.as_mut().set_status_message(QString::from(&format!("Model saved to {}", path)));
            }
            Err(e) => {
                self.as_mut().set_status_message(QString::from(&format!("Save error: {}", e)));
            }
        }
    }

    fn load_model(mut self: Pin<&mut Self>, filename: QString) {
        let path = filename.to_string();
        match CudaGnnFacade::from_model_file(&path) {
            Ok(facade) => {
                let fs = facade.get_feature_size() as i32;
                let hs = facade.get_hidden_size() as i32;
                let os = facade.get_output_size() as i32;
                let mp = facade.get_num_message_passing_layers() as i32;
                let lr = facade.get_learning_rate() as f64;

                self.as_mut().set_feature_size(fs);
                self.as_mut().set_hidden_size(hs);
                self.as_mut().set_output_size(os);
                self.as_mut().set_num_mp_layers(mp);
                self.as_mut().set_learning_rate(lr);
                self.as_mut().rust_mut().facade = Some(facade);
                self.as_mut().set_network_created(true);
                self.as_mut().set_status_message(QString::from(&format!("Model loaded from {}", path)));
            }
            Err(e) => {
                self.as_mut().set_status_message(QString::from(&format!("Load error: {}", e)));
            }
        }
    }

    fn get_node_feature(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;
        let feature = *self.as_ref().facade_feature_idx() as usize;

        let value = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_node_feature(node, feature)
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!(
            "Node[{}].feature[{}] = {}",
            node, feature, value
        )));
    }

    fn get_node_features(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let features_str = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            if let Some(features) = facade.get_node_features(node) {
                features.iter().map(|f| format!("{:.4}", f)).collect::<Vec<_>>().join(",\n  ")
            } else {
                return;
            }
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!(
            "Node[{}].features = [\n  {}\n]",
            node, features_str
        )));
    }

    fn get_edge_features(mut self: Pin<&mut Self>) {
        let edge = *self.as_ref().facade_edge_idx() as usize;

        let features_str = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            if let Some(features) = facade.get_edge_features(edge) {
                features.iter().map(|f| format!("{:.4}", f)).collect::<Vec<_>>().join(", ")
            } else {
                "[]".to_string()
            }
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!(
            "Edge[{}].features = [{}]",
            edge, features_str
        )));
    }

    fn get_num_nodes_facade(mut self: Pin<&mut Self>) {
        let count = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_num_nodes()
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Number of nodes = {}", count)));
    }

    fn get_num_edges_facade(mut self: Pin<&mut Self>) {
        let count = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_num_edges()
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Number of edges = {}", count)));
    }

    fn get_neighbors(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let neighbors_str = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            if let Some(neighbors) = facade.get_neighbors(node) {
                neighbors.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(", ")
            } else {
                return;
            }
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!(
            "Node[{}].neighbors = [{}]",
            node, neighbors_str
        )));
    }

    fn get_adjacency_matrix(mut self: Pin<&mut Self>) {
        let output = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            let n = facade.get_num_nodes();
            let mut matrix = vec![vec![0; n]; n];

            for i in 0..n {
                if let Some(neighbors) = facade.get_neighbors(i) {
                    for &j in neighbors {
                        if j < n {
                            matrix[i][j] = 1;
                        }
                    }
                }
            }

            let mut s = "Adjacency Matrix:\n".to_string();
            for row in &matrix {
                s.push_str(&format!("  [{:?}]\n", row));
            }
            s
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&output));
    }

    fn get_edge_endpoints(mut self: Pin<&mut Self>) {
        let edge = *self.as_ref().facade_edge_idx() as usize;

        let output = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            if let Some((src, tgt)) = facade.get_edge_endpoints(edge) {
                format!("Edge[{}] = {} -> {}", edge, src, tgt)
            } else {
                format!("Edge[{}] = not found", edge)
            }
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&output));
    }

    fn has_edge(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;
        let neighbor = *self.as_ref().facade_neighbor_idx() as usize;

        let result = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.has_edge(node, neighbor)
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!(
            "Has edge {} -> {}? {}",
            node, neighbor, result
        )));
    }

    fn get_in_degree(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let deg = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_in_degree(node)
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Node[{}].inDegree = {}", node, deg)));
    }

    fn get_out_degree(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let deg = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_out_degree(node)
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Node[{}].outDegree = {}", node, deg)));
    }

    fn get_graph_embedding(mut self: Pin<&mut Self>) {
        let emb_str = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            let emb = facade.get_graph_embedding();
            emb.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", ")
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Graph embedding:\n[{}]", emb_str)));
    }

    fn get_node_mask(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let mask = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_node_mask(node)
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Node[{}].mask = {}", node, mask)));
    }

    fn toggle_node_mask(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let new_mask = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            let current = facade.get_node_mask(node);
            facade.set_node_mask(node, !current);
            !current
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Node[{}].mask toggled to {}", node, new_mask)));
    }

    fn apply_node_dropout(mut self: Pin<&mut Self>) {
        let count = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.apply_node_dropout(0.3);
            facade.get_masked_node_count()
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Applied 30% node dropout. Active nodes: {}", count)));
    }

    fn apply_edge_dropout(mut self: Pin<&mut Self>) {
        let count = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.apply_edge_dropout(0.3);
            facade.get_masked_edge_count()
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Applied 30% edge dropout. Active edges: {}", count)));
    }

    fn get_masked_counts(mut self: Pin<&mut Self>) {
        let (nodes, edges) = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            (facade.get_masked_node_count(), facade.get_masked_edge_count())
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Active nodes: {}\nActive edges: {}", nodes, edges)));
    }

    fn add_node(mut self: Pin<&mut Self>, features: QString) {
        let features_vec: Vec<f32> = features.to_string()
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let (idx, total) = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            let idx = facade.add_node(features_vec);
            (idx, facade.get_num_nodes())
        } else {
            return;
        };

        self.as_mut().set_num_nodes(total as i32);
        self.as_mut().set_facade_output(QString::from(&format!("Added node at index {}. Total nodes: {}", idx, total)));
    }

    fn remove_node(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let (total_nodes, total_edges) = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.remove_node(node);
            (facade.get_num_nodes(), facade.get_num_edges())
        } else {
            return;
        };

        self.as_mut().set_num_nodes(total_nodes as i32);
        self.as_mut().set_num_edges(total_edges as i32);
        self.as_mut().set_facade_output(QString::from(&format!("Removed node {}. Total nodes: {}", node, total_nodes)));
    }

    fn add_edge_facade(mut self: Pin<&mut Self>) {
        let src = *self.as_ref().facade_node_idx() as usize;
        let tgt = *self.as_ref().facade_neighbor_idx() as usize;

        let (idx, total) = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            let idx = facade.add_edge(src, tgt, vec![]);
            (idx, facade.get_num_edges())
        } else {
            return;
        };

        self.as_mut().set_num_edges(total as i32);
        self.as_mut().set_facade_output(QString::from(&format!("Added edge {} -> {} at index {}. Total edges: {}", src, tgt, idx, total)));
    }

    fn remove_edge(mut self: Pin<&mut Self>) {
        let edge = *self.as_ref().facade_edge_idx() as usize;

        let total = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.remove_edge(edge);
            facade.get_num_edges()
        } else {
            return;
        };

        self.as_mut().set_num_edges(total as i32);
        self.as_mut().set_facade_output(QString::from(&format!("Removed edge {}. Total edges: {}", edge, total)));
    }

    fn clear_all_edges(mut self: Pin<&mut Self>) {
        if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.clear_all_edges();
        } else {
            return;
        }

        self.as_mut().set_num_edges(0);
        self.as_mut().set_facade_output(QString::from("Cleared all edges"));
    }

    fn rebuild_adjacency(mut self: Pin<&mut Self>) {
        if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.rebuild_adjacency_list();
        } else {
            return;
        }

        self.as_mut().set_facade_output(QString::from("Rebuilt adjacency list"));
    }

    fn get_node_degree(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;

        let (in_deg, out_deg) = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            (facade.get_in_degree(node), facade.get_out_degree(node))
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!(
            "Node[{}].degree = {} (in: {}, out: {})",
            node, in_deg + out_deg, in_deg, out_deg
        )));
    }

    fn compute_page_rank(mut self: Pin<&mut Self>) {
        let output = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            let ranks = facade.compute_page_rank(0.85, 20);
            let mut s = "PageRank:\n".to_string();
            for (i, r) in ranks.iter().enumerate() {
                s.push_str(&format!("  Node {}: {:.4}\n", i, r));
            }
            s
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&output));
    }

    fn get_parameter_count(mut self: Pin<&mut Self>) {
        let count = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_parameter_count()
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&format!("Total parameters = {}", count)));
    }

    fn get_architecture_summary(mut self: Pin<&mut Self>) {
        let summary = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.get_architecture_summary()
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&summary));
    }

    fn export_graph_json(mut self: Pin<&mut Self>) {
        let json = if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            facade.export_graph_to_json()
        } else {
            return;
        };

        self.as_mut().set_facade_output(QString::from(&json));
    }

    fn set_node_feature(mut self: Pin<&mut Self>) {
        let node = *self.as_ref().facade_node_idx() as usize;
        let feature = *self.as_ref().facade_feature_idx() as usize;
        let value = *self.as_ref().facade_set_value() as f32;

        if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.set_node_feature(node, feature, value);
        } else {
            return;
        }

        self.as_mut().set_facade_output(QString::from(&format!("Set Node[{}].feature[{}] = {}", node, feature, value)));
    }

    fn set_learning_rate_facade(mut self: Pin<&mut Self>) {
        let value = *self.as_ref().facade_set_value() as f32;

        if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.set_learning_rate(value);
        } else {
            return;
        }

        self.as_mut().set_learning_rate(value as f64);
        self.as_mut().set_facade_output(QString::from(&format!("Set learning rate = {}", value)));
    }

    fn get_graph_data(mut self: Pin<&mut Self>) -> QString {
        if let Some(facade) = self.as_ref().rust().facade.as_ref() {
            QString::from(&facade.export_graph_to_json())
        } else {
            QString::from("{}")
        }
    }

    fn import_from_csv(mut self: Pin<&mut Self>, nodes_csv: QString, edges_csv: QString) {
        let nodes_text = nodes_csv.to_string();
        let edges_text = edges_csv.to_string();

        let mut node_features: Vec<Vec<f32>> = Vec::new();
        for line in nodes_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<f32> = line
                .split(',')
                .skip(1)
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !parts.is_empty() {
                node_features.push(parts);
            }
        }

        let mut edges: Vec<(usize, usize)> = Vec::new();
        for line in edges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                if let (Ok(s), Ok(t)) = (
                    parts[0].trim().parse::<usize>(),
                    parts[1].trim().parse::<usize>(),
                ) {
                    edges.push((s, t));
                }
            }
        }

        if node_features.is_empty() {
            self.as_mut().set_status_message(QString::from("No valid node data found in CSV"));
            return;
        }

        let num_nodes = node_features.len();
        let feature_size = node_features[0].len();

        let num_edges = if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            facade.create_empty_graph(num_nodes, feature_size);

            for (i, features) in node_features.iter().enumerate() {
                facade.set_node_features(i, features.clone());
            }

            for (s, t) in edges {
                if s < num_nodes && t < num_nodes {
                    facade.add_edge(s, t, vec![]);
                }
            }

            facade.rebuild_adjacency_list();
            facade.get_num_edges()
        } else {
            return;
        };

        self.as_mut().set_num_nodes(num_nodes as i32);
        self.as_mut().set_num_edges(num_edges as i32);
        self.as_mut().set_feature_size(feature_size as i32);
        self.as_mut().set_graph_loaded(true);
        self.as_mut().set_status_message(QString::from(&format!(
            "Imported {} nodes with {} features, {} edges",
            num_nodes, feature_size, num_edges
        )));
    }

    fn apply_manual_features(mut self: Pin<&mut Self>, features_text: QString) {
        let text = features_text.to_string();

        if let Some(facade) = self.as_mut().rust_mut().facade.as_mut() {
            for (i, line) in text.lines().enumerate() {
                let features: Vec<f32> = line
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if !features.is_empty() {
                    facade.set_node_features(i, features);
                }
            }
        } else {
            return;
        }

        self.as_mut().set_status_message(QString::from("Applied manual node features"));
    }
}
