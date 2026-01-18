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

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import com.glassboxai.gnn 1.0

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1200
    height: 900
    title: "Facaded GNN - CUDA Accelerated"
    color: "#fafafa"

    GnnBridge {
        id: gnnBridge
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: 20

        ColumnLayout {
            width: mainWindow.width - 60
            spacing: 20

            Text {
                text: "Facaded GNN"
                font.pixelSize: 28
                font.bold: true
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "Network Configuration"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Feature Size:" }
                            SpinBox {
                                id: featureSizeInput
                                from: 1
                                to: 100
                                value: gnnBridge.feature_size
                                onValueChanged: gnnBridge.feature_size = value
                            }
                        }

                        RowLayout {
                            Label { text: "Hidden Size:" }
                            SpinBox {
                                id: hiddenSizeInput
                                from: 4
                                to: 256
                                value: gnnBridge.hidden_size
                                onValueChanged: gnnBridge.hidden_size = value
                            }
                        }

                        RowLayout {
                            Label { text: "Output Size:" }
                            SpinBox {
                                id: outputSizeInput
                                from: 1
                                to: 100
                                value: gnnBridge.output_size
                                onValueChanged: gnnBridge.output_size = value
                            }
                        }

                        RowLayout {
                            Label { text: "MP Layers:" }
                            SpinBox {
                                id: mpLayersInput
                                from: 1
                                to: 8
                                value: gnnBridge.num_mp_layers
                                onValueChanged: gnnBridge.num_mp_layers = value
                            }
                        }
                    }

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Activation:" }
                            ComboBox {
                                id: activationCombo
                                model: ["relu", "leaky_relu", "tanh", "sigmoid"]
                                currentIndex: 0
                                onCurrentTextChanged: gnnBridge.activation = currentText
                            }
                        }

                        RowLayout {
                            Label { text: "Loss Function:" }
                            ComboBox {
                                id: lossFunctionCombo
                                model: ["mse", "bce"]
                                currentIndex: 0
                                onCurrentTextChanged: gnnBridge.loss_function = currentText
                            }
                        }

                        RowLayout {
                            Label { text: "Learning Rate:" }
                            TextField {
                                id: learningRateInput
                                text: gnnBridge.learning_rate.toFixed(4)
                                implicitWidth: 80
                                validator: DoubleValidator { bottom: 0.0001; top: 1.0 }
                                onEditingFinished: gnnBridge.learning_rate = parseFloat(text)
                            }
                        }
                    }

                    RowLayout {
                        Button {
                            text: "Create Network"
                            onClicked: gnnBridge.create_network()
                        }
                        Label {
                            text: gnnBridge.status_message
                            color: "#666"
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "Graph Configuration"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Number of Nodes:" }
                            SpinBox {
                                id: numNodesInput
                                from: 2
                                to: 100
                                value: gnnBridge.num_nodes
                                onValueChanged: gnnBridge.num_nodes = value
                            }
                        }

                        CheckBox {
                            id: undirectedCheck
                            text: "Undirected"
                            checked: gnnBridge.undirected
                            onCheckedChanged: gnnBridge.undirected = checked
                        }

                        CheckBox {
                            id: selfLoopsCheck
                            text: "Self-Loops"
                            checked: gnnBridge.self_loops
                            onCheckedChanged: gnnBridge.self_loops = checked
                        }
                    }

                    Label { text: "Edges (source,target per line):" }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 120

                        TextArea {
                            id: edgeListInput
                            text: gnnBridge.edge_list
                            font.family: "monospace"
                            wrapMode: TextArea.Wrap
                            onTextChanged: gnnBridge.edge_list = text
                        }
                    }

                    RowLayout {
                        Button {
                            text: "Build Graph"
                            onClicked: gnnBridge.build_graph()
                        }
                        Button {
                            text: "Randomize Features"
                            onClicked: gnnBridge.randomize_features()
                        }
                        Label {
                            text: gnnBridge.graph_loaded ? 
                                  "Graph: " + gnnBridge.num_nodes + " nodes, " + gnnBridge.num_edges + " edges" : 
                                  "No graph loaded"
                            color: "#666"
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "Graph Visualization"
                Layout.fillWidth: true
                Layout.preferredHeight: 350

                ColumnLayout {
                    anchors.fill: parent

                    Canvas {
                        id: graphCanvas
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 300

                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        
                        if (!gnnBridge.graph_loaded) {
                            ctx.fillStyle = "#ccc"
                            ctx.font = "14px sans-serif"
                            ctx.fillText("Build graph to see visualization", 10, 30)
                            return
                        }

                        var graphData
                        try {
                            graphData = JSON.parse(gnnBridge.get_graph_data())
                        } catch (e) {
                            return
                        }

                        var numNodes = graphData.numNodes
                        var positions = []
                        var centerX = width / 2
                        var centerY = height / 2
                        var radius = Math.min(width, height) * 0.35

                        for (var i = 0; i < numNodes; i++) {
                            var angle = (2 * Math.PI * i) / numNodes - Math.PI / 2
                            positions.push({
                                x: centerX + radius * Math.cos(angle),
                                y: centerY + radius * Math.sin(angle)
                            })
                        }

                        ctx.strokeStyle = "#94a3b8"
                        ctx.lineWidth = 2

                        for (var j = 0; j < graphData.edges.length; j++) {
                            var edge = graphData.edges[j]
                            var p1 = positions[edge.source]
                            var p2 = positions[edge.target]
                            
                            ctx.beginPath()
                            ctx.moveTo(p1.x, p1.y)
                            ctx.lineTo(p2.x, p2.y)
                            ctx.stroke()
                        }

                        for (var k = 0; k < numNodes; k++) {
                            var pos = positions[k]
                            var node = graphData.nodes[k]
                            
                            var normMag = 0.5
                            var r = Math.round(68 + normMag * 120)
                            var g = Math.round(119 - normMag * 50)
                            var b = Math.round(200 - normMag * 100)
                            
                            ctx.fillStyle = node.masked ? "rgb(" + r + "," + g + "," + b + ")" : "#ccc"
                            ctx.strokeStyle = "#1e293b"
                            ctx.lineWidth = 2
                            
                            ctx.beginPath()
                            ctx.arc(pos.x, pos.y, 18, 0, Math.PI * 2)
                            ctx.fill()
                            ctx.stroke()
                            
                            ctx.fillStyle = "white"
                            ctx.font = "bold 11px sans-serif"
                            ctx.textAlign = "center"
                            ctx.textBaseline = "middle"
                            ctx.fillText(k.toString(), pos.x, pos.y)
                        }
                    }

                    Connections {
                        target: gnnBridge
                        function onGraph_loadedChanged() {
                            graphCanvas.requestPaint()
                        }
                        function onNum_nodesChanged() {
                            graphCanvas.requestPaint()
                        }
                        function onNum_edgesChanged() {
                            graphCanvas.requestPaint()
                        }
                    }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "Training"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        spacing: 20

                        RowLayout {
                            Label { text: "Target Output (comma separated):" }
                            TextField {
                                id: targetOutputInput
                                text: gnnBridge.target_output
                                implicitWidth: 200
                                onTextChanged: gnnBridge.target_output = text
                            }
                        }

                        RowLayout {
                            Label { text: "Training Iterations:" }
                            SpinBox {
                                id: trainItersInput
                                from: 1
                                to: 2000
                                value: gnnBridge.train_iterations
                                onValueChanged: gnnBridge.train_iterations = value
                            }
                        }
                    }

                    RowLayout {
                        Button {
                            text: "Train"
                            onClicked: gnnBridge.train_network()
                        }
                        Button {
                            text: "Predict Only"
                            onClicked: gnnBridge.predict_only()
                        }
                    }

                    Label {
                        text: gnnBridge.current_loss > 0 ? "Loss: " + gnnBridge.current_loss.toFixed(6) : ""
                        color: "#666"
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 60
                        color: "#f5f5f5"
                        radius: 5
                        visible: gnnBridge.predict_output.length > 0

                        Text {
                            anchors.fill: parent
                            anchors.margins: 10
                            text: gnnBridge.predict_output
                            font.family: "monospace"
                            wrapMode: Text.Wrap
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "Model Persistence"
                Layout.fillWidth: true

                RowLayout {
                    spacing: 10

                    Button {
                        text: "Save Model"
                        onClicked: saveDialog.open()
                    }

                    Button {
                        text: "Load Model"
                        onClicked: loadDialog.open()
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "🔧 GNN Facade API Explorer"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 15

                    Text {
                        text: "Use the facade to inspect and modify the GNN internals:"
                        color: "#666"
                    }

                    GridLayout {
                        columns: 7
                        rowSpacing: 5
                        columnSpacing: 10

                        Label { text: "Node Idx / Edge Src:" }
                        SpinBox {
                            id: facadeNodeIdx
                            from: 0
                            to: 99
                            value: gnnBridge.facade_node_idx
                            onValueChanged: gnnBridge.facade_node_idx = value
                        }

                        Label { text: "Edge Idx:" }
                        SpinBox {
                            id: facadeEdgeIdx
                            from: 0
                            to: 999
                            value: gnnBridge.facade_edge_idx
                            onValueChanged: gnnBridge.facade_edge_idx = value
                        }

                        Label { text: "Layer Idx:" }
                        SpinBox {
                            id: facadeLayerIdx
                            from: 0
                            to: 10
                            value: gnnBridge.facade_layer_idx
                            onValueChanged: gnnBridge.facade_layer_idx = value
                        }

                        Item { Layout.fillWidth: true }

                        Label { text: "Feature Idx:" }
                        SpinBox {
                            id: facadeFeatureIdx
                            from: 0
                            to: 99
                            value: gnnBridge.facade_feature_idx
                            onValueChanged: gnnBridge.facade_feature_idx = value
                        }

                        Label { text: "Edge Tgt:" }
                        SpinBox {
                            id: facadeNeighborIdx
                            from: 0
                            to: 99
                            value: gnnBridge.facade_neighbor_idx
                            onValueChanged: gnnBridge.facade_neighbor_idx = value
                        }

                        Label { text: "Value:" }
                        TextField {
                            id: facadeSetValue
                            text: gnnBridge.facade_set_value.toFixed(4)
                            implicitWidth: 80
                            onEditingFinished: gnnBridge.facade_set_value = parseFloat(text)
                        }

                        Item { Layout.fillWidth: true }
                    }

                    Text {
                        text: "1. Node & Edge Features"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Get Node Feature"; onClicked: gnnBridge.get_node_feature() }
                        Button { text: "Get All Node Features"; onClicked: gnnBridge.get_node_features() }
                        Button { text: "Get Edge Features"; onClicked: gnnBridge.get_edge_features() }
                        Button { text: "Get Num Nodes"; onClicked: gnnBridge.get_num_nodes_facade() }
                        Button { text: "Get Num Edges"; onClicked: gnnBridge.get_num_edges_facade() }
                    }

                    Text {
                        text: "2. Topology"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Get Neighbors"; onClicked: gnnBridge.get_neighbors() }
                        Button { text: "Get Adjacency Matrix"; onClicked: gnnBridge.get_adjacency_matrix() }
                        Button { text: "Get Edge Endpoints"; onClicked: gnnBridge.get_edge_endpoints() }
                        Button { text: "Has Edge?"; onClicked: gnnBridge.has_edge() }
                        Button { text: "Get In-Degree"; onClicked: gnnBridge.get_in_degree() }
                        Button { text: "Get Out-Degree"; onClicked: gnnBridge.get_out_degree() }
                    }

                    Text {
                        text: "3. Embeddings"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Get Graph Embedding"; onClicked: gnnBridge.get_graph_embedding() }
                    }

                    Text {
                        text: "4. Masking & Dropout"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Get Node Mask"; onClicked: gnnBridge.get_node_mask() }
                        Button { text: "Toggle Node Mask"; onClicked: gnnBridge.toggle_node_mask() }
                        Button { text: "Apply Node Dropout (30%)"; onClicked: gnnBridge.apply_node_dropout() }
                        Button { text: "Apply Edge Dropout (30%)"; onClicked: gnnBridge.apply_edge_dropout() }
                        Button { text: "Get Masked Counts"; onClicked: gnnBridge.get_masked_counts() }
                    }

                    Text {
                        text: "5. Graph Mutation"
                        font.bold: true
                    }

                    RowLayout {
                        Label { text: "New Features:" }
                        TextField {
                            id: newNodeFeatures
                            text: "0.5,0.5,0.5"
                            implicitWidth: 150
                        }
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Add Node"; onClicked: gnnBridge.add_node(newNodeFeatures.text) }
                        Button { text: "Remove Node"; onClicked: gnnBridge.remove_node() }
                        Button { text: "Add Edge"; onClicked: gnnBridge.add_edge_facade() }
                        Button { text: "Remove Edge"; onClicked: gnnBridge.remove_edge() }
                        Button { text: "Clear All Edges"; onClicked: gnnBridge.clear_all_edges() }
                        Button { text: "Rebuild Adjacency"; onClicked: gnnBridge.rebuild_adjacency() }
                    }

                    Text {
                        text: "6. Diagnostics & Centrality"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Get Node Degree"; onClicked: gnnBridge.get_node_degree() }
                        Button { text: "Compute PageRank"; onClicked: gnnBridge.compute_page_rank() }
                    }

                    Text {
                        text: "7. Architecture"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Get Parameter Count"; onClicked: gnnBridge.get_parameter_count() }
                        Button { text: "Get Architecture Summary"; onClicked: gnnBridge.get_architecture_summary() }
                    }

                    Text {
                        text: "8. Export"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Export Graph JSON"; onClicked: gnnBridge.export_graph_json() }
                    }

                    Text {
                        text: "Setters"
                        font.bold: true
                    }

                    Flow {
                        Layout.fillWidth: true
                        spacing: 5

                        Button { text: "Set Node Feature"; onClicked: gnnBridge.set_node_feature() }
                        Button { text: "Set Learning Rate"; onClicked: gnnBridge.set_learning_rate_facade() }
                    }

                    Text {
                        text: "Output"
                        font.bold: true
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 200
                        color: "white"
                        border.color: "#ccc"
                        border.width: 1

                        ScrollView {
                            anchors.fill: parent
                            anchors.margins: 5

                            TextArea {
                                id: facadeOutputText
                                text: gnnBridge.facade_output
                                font.family: "monospace"
                                readOnly: true
                                wrapMode: TextArea.Wrap
                                background: null
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "Import Data from CSV"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Text {
                        text: "Import node features and edges from CSV data"
                        color: "#666"
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 20

                        ColumnLayout {
                            Layout.fillWidth: true

                            Text {
                                text: "Node Features CSV"
                                font.bold: true
                            }
                            Text {
                                text: "Format: nodeId,feature1,feature2,..."
                                color: "#888"
                                font.pixelSize: 11
                            }

                            ScrollView {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 100

                                TextArea {
                                    id: nodesCsvText
                                    font.family: "monospace"
                                    placeholderText: "0,0.5,0.3,0.2\n1,0.1,0.8,0.4\n2,0.9,0.2,0.6"
                                }
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true

                            Text {
                                text: "Edges CSV"
                                font.bold: true
                            }
                            Text {
                                text: "Format: source,target"
                                color: "#888"
                                font.pixelSize: 11
                            }

                            ScrollView {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 100

                                TextArea {
                                    id: edgesCsvText
                                    font.family: "monospace"
                                    placeholderText: "0,1\n1,2\n2,0"
                                }
                            }
                        }
                    }

                    RowLayout {
                        Button {
                            text: "Import from CSV"
                            onClicked: gnnBridge.import_from_csv(nodesCsvText.text, edgesCsvText.text)
                        }
                        Button {
                            text: "Clear CSV Inputs"
                            onClicked: {
                                nodesCsvText.text = ""
                                edgesCsvText.text = ""
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#ccc"
            }

            GroupBox {
                title: "Manual Node Features Input"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Text {
                        text: "Enter node features manually for the current graph (one node per line)"
                        color: "#666"
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 120

                        TextArea {
                            id: manualNodeFeatures
                            font.family: "monospace"
                            placeholderText: "0.5,0.3,0.2\n0.1,0.8,0.4\n0.9,0.2,0.6\n0.4,0.5,0.1\n0.7,0.3,0.8"
                        }
                    }

                    RowLayout {
                        Button {
                            text: "Apply Node Features"
                            onClicked: gnnBridge.apply_manual_features(manualNodeFeatures.text)
                        }
                    }
                }
            }

            Item {
                Layout.fillWidth: true
                height: 40
            }
        }
    }

    FileDialog {
        id: saveDialog
        title: "Save Model"
        selectExisting: false
        nameFilters: ["Model files (*.bin)", "All files (*)"]
        onAccepted: gnnBridge.save_model(fileUrl)
    }

    FileDialog {
        id: loadDialog
        title: "Load Model"
        selectExisting: true
        nameFilters: ["Model files (*.bin)", "All files (*)"]
        onAccepted: gnnBridge.load_model(fileUrl)
    }
}
