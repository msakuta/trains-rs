//! Dijkstra pathfinding

use std::collections::{BinaryHeap, HashMap};

use crate::train_tracks::{ConnectPoint, NodeConnection, PathConnection, TrainTracks};

use super::Train;

#[derive(Clone, Copy, Debug)]
struct SearchNode {
    node_con: NodeConnection,
    came_from: Option<PathConnection>,
    cost: f64,
}

impl std::cmp::PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl std::cmp::Eq for SearchNode {}

impl std::cmp::PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl std::cmp::Ord for SearchNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.partial_cmp(&other.cost).unwrap()
    }
}

impl Train {
    pub(super) fn follow_route(&mut self, tracks: &TrainTracks) {
        if self.route.is_empty() {
            return;
        }

        let Some(path_con) = self.route.last() else {
            return;
        };

        if path_con.path_id == self.path_id {
            if let Some(path) = tracks.paths.get(&path_con.path_id) {
                if let Some(node) = tracks
                    .nodes
                    .get(&path.connect_node(path_con.connect_point).node_id)
                {
                    if let Some(idx) = node
                        .paths_in_direction(match path_con.connect_point {
                            ConnectPoint::Start => !path.start_node.direction,
                            ConnectPoint::End => !path.end_node.direction,
                        })
                        .iter()
                        .enumerate()
                        .find(|(_, p)| p.path_id == self.path_id)
                    {
                        self.switch_path = idx.0;
                    }
                }
                match path_con.connect_point {
                    ConnectPoint::Start => {
                        self.move_to_s(0.);
                    }
                    ConnectPoint::End => {
                        self.move_to_s(path.s_length());
                    }
                }
            }
        } else {
            self.route.pop();
        }
    }

    pub(super) fn find_path(&mut self, path_id: usize, tracks: &TrainTracks) {
        if !self.route.is_empty() {
            return;
        }

        let Some(self_path) = tracks.paths.get(&self.path_id) else {
            return;
        };

        let start_node = SearchNode {
            node_con: self_path.start_node,
            came_from: Some(PathConnection {
                path_id: self.path_id,
                connect_point: ConnectPoint::Start,
            }),
            cost: self.s,
        };

        let end_node = SearchNode {
            node_con: self_path.end_node,
            came_from: Some(PathConnection {
                path_id: self.path_id,
                connect_point: ConnectPoint::End,
            }),
            cost: self_path.s_length() - self.s,
        };

        let mut closed: HashMap<NodeConnection, SearchNode> = HashMap::new();
        closed.insert(self_path.start_node, start_node);
        closed.insert(self_path.end_node, end_node);

        let mut queue: BinaryHeap<SearchNode> = BinaryHeap::new();
        queue.push(start_node);
        queue.push(end_node);

        println!("Starting search from path {:?}", self.path_id);

        while let Some(current) = queue.pop() {
            let Some(node) = tracks.nodes.get(&current.node_con.node_id) else {
                continue;
            };
            println!("Searching path through node {:?}", current.node_con);

            for &path_con in node.paths_in_direction(!current.node_con.direction) {
                println!("Searching path through path {path_con:?}");
                if path_con.path_id == path_id {
                    println!("Found path to path:{path_id}");
                    let mut nodes = vec![path_con];
                    let mut con = current;
                    loop {
                        let Some(came_from) = con.came_from else {
                            break;
                        };
                        nodes.push(came_from);
                        if came_from.path_id == self.path_id {
                            break;
                        }
                        let Some(path) = tracks.paths.get(&came_from.path_id) else {
                            break;
                        };
                        let Some(next_con) =
                            closed.get(&path.connect_node(!came_from.connect_point))
                        else {
                            break;
                        };
                        println!("  from {:?}", next_con);
                        con = *next_con;
                    }
                    nodes.reverse();
                    println!("Route: {:?}", nodes);
                    self.route = nodes;
                    return;
                }

                let Some(path) = tracks.paths.get(&path_con.path_id) else {
                    continue;
                };

                let next_node = path.connect_node(path_con.connect_point);

                let existing_state = closed.entry(next_node).or_insert_with(|| SearchNode {
                    node_con: next_node,
                    came_from: None,
                    cost: f64::INFINITY,
                });
                let new_cost = current.cost + path.s_length();
                if current.cost < existing_state.cost {
                    existing_state.cost = new_cost;
                    existing_state.came_from = Some(path_con);

                    queue.push(SearchNode {
                        node_con: next_node,
                        came_from: Some(path_con),
                        cost: new_cost,
                    });
                }
            }
        }
    }
}
