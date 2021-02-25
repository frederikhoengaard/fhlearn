import numpy as np
import random
import heapq



class Node:
    def __init__(
            self, 
            features: np.array, 
            labels: np.array
        ):                            
        self.features = features
        self.labels = labels
        self.leftChild = None
        self.rightChild = None
        self.depth: int = None
        self.gini: float = None
        self.split_feature = None
        self.split_threshold = None
        self.is_leaf: bool = None
        self.majority_class = None
        self.class_probabilities: dict = None
        self.n_obs: int = None

        
        
class DecisionTreeClassifier:
    def __init__(
            self, 
            max_depth: int = float('inf'), 
            min_samples_split: int = 2
        ):
        self.root: Node = None
        self.tree_depth: int = 0
        self.n_leaf_nodes: int = 0
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = None
        self.min_weight_fraction: float = None
        self.max_leaf_nodes: int = None
        self.max_features: int = None
        self.random_state: int = None   
        self.nodes_total = 0

            
            
    def _number_of_classes(self, labels: np.array) -> int:
        return len(np.unique(labels))
    
    
    
    def _get_n_obs(self, data: np.array) -> int:
        return np.shape(data)[0]
  


    def _get_classes(self, labels: np.array) -> list:
        return list(set(labels))

    
    
    def _get_n_features(self, features: np.array) -> int:
        return np.shape(features)[1]
  


    def _n_class_occurence(self, labels: np.array) -> dict:
        counts = {}
        for obs in labels:
            if obs not in counts:
                counts[obs] = 1
            else:
                counts[obs] += 1     
        return counts

    
    
    def calc_gini(self, labels: np.array) -> float:
        gini = 1
        occurrences = self._n_class_occurence(labels)
        for classname in self._get_classes(labels):
            gini -= (occurrences[classname] / len(labels)) ** 2
        return gini 
    
    
    
    def get_best_carts(self, scores: list) -> list:
        if not list:
            return ValueError('No CART scores supplied')
        best = min(scores)[0]
        best_scores = []
        for score in scores:
            if score[0] == best:
                best_scores.append(score)
            else:
                break
        return best_scores
    


    def _get_majority_class(self, labels: np.array):
        if self.random_state:
            random.seed(self.random_state)
        major_classes = []
        max_count = float('-inf')
        for key,count in self._n_class_occurence(labels).items():
            if count > max_count:
                max_count = count
                major_classes = [key]
            elif count == max_count:
                major_classes.append(key)
            else:
                continue
        return random.choice(major_classes)
    


    def _get_class_probabilities(self, labels: np.array) -> dict:
        probabilities = {}
        n_obs = self._get_n_obs(labels)
        for key,count in self._n_class_occurence(labels).items():
            probabilities[key] = count / n_obs
        return probabilities

    
    
    def _calc_CART(
            self, 
            left_labels: np.array, 
            right_labels: np.array
        ) -> float: 
        
        """
        Calculates CART as defined in HML p. 171, eq 6.2
        
        J(k,t_k) = (m_left / m) * G_left + (m_right / m) * G_right

        where m refers to number of samples and G to gini score
        """
        
        left_gini, right_gini = self.calc_gini(left_labels), self.calc_gini(right_labels)
        m_left, m_right = self._get_n_obs(left_labels), self._get_n_obs(right_labels)
        m = m_left + m_right
        return (m_left / m) * left_gini + (m_right / m) * right_gini
    


    def _find_next_feature_val(
            self,
            features,
            feature_index,
            val
        ) -> float:
        relevant_features = features[:,feature_index]
        relevant_features.sort()
        found_val = False
        for i in range(len(relevant_features)):
            if not found_val:
                if relevant_features[i] == val:
                    found_val = True
            if found_val:
                if relevant_features[i] > val:
                    return relevant_features[i]
        return val
     


    def _split_data(
            self, 
            features: np.array, 
            labels: np.array, 
            feature_index: int, 
            threshold: float
        ) -> list:
        data = features[:,feature_index]
        mask = data <= threshold
        left = np.c_[features,labels][mask]
        right = np.c_[features,labels][~mask]
        return [left,right]



    def _find_best_split(
            self, 
            features: np.array, 
            labels: np.array
        ) -> list: 
        if self.random_state:
            random.seed(self.random_state)
        cart_scores = []
        for col in range(self._get_n_features(features)):
            thresholds = features[:,col]
            for threshold in thresholds:
                left, right = self._split_data(features,labels,col,threshold)
                CART_score = self._calc_CART(left[:,-1], right[:,-1])
                heapq.heappush(cart_scores, (CART_score,col,threshold))
        cart_scores = self.get_best_carts(cart_scores)
        choice = random.choice(cart_scores)
        pos_split_val = ((self._find_next_feature_val(np.copy(features),choice[1],choice[2]) - choice[2]) / 2) + choice[2]
        return [choice[0],choice[1],pos_split_val]



    def _passes_hyperparameters(self, node: Node) -> bool:
        if node.depth < self.max_depth and node.n_obs >= self.min_samples_split and node.gini != 0:
            return True
        return False



    def _decide_if_leaf(self, node):
        if not self._passes_hyperparameters(node):
            return True
        return False



     def _insert_node(
            self, 
            features: np.array, 
            labels: np.array, 
            depth: int
        ) -> Node:
        node = Node(features,labels)
        node.depth = depth
        node.gini = self.calc_gini(labels)
        node.class_probabilities = self._get_class_probabilities
        node.n_obs = self._get_n_obs(labels)
        node.is_leaf = self._decide_if_leaf(node)
        node.majority_class = self._get_majority_class(labels)
        if not node.is_leaf: # means we will try splitting
            gini,feature,threshold = self._find_best_split(features,labels)
            if gini == node.gini:                
                node.is_leaf = True
                self.n_leaf_nodes += 1
                return node # cannot make better split than before, so creating leaf
            node.split_feature,node.split_threshold = feature,threshold
            left,right = self._split_data(features,labels,feature,threshold)
            node.leftChild = self._insert_node(left[:,:-1], left[:,-1], depth + 1)
            node.rightChild = self._insert_node(right[:,:-1], right[:,-1], depth + 1)
            return node
        else: 
            self.n_leaf_nodes += 1
            return node
        


    def fit(
            self, 
            features: np.array, 
            labels: np.array, 
            criterion='gini'
        ):
        self.root = self._insert_node(features,labels,self.tree_depth)

    

    def _predict_sample(
            self,
            sample: np.array, 
            node: Node
        ):
        if node.is_leaf:
            return node.majority_class
        else:
            sample_val = sample[node.split_feature]
            if sample_val <= node.split_threshold:
                return self._predict_sample(sample,node.leftChild)
            else:
                return self._predict_sample(sample,node.rightChild)



    def predict(self, features: np.array) -> np.array:
        predictions = []
        for obs in range(self._get_n_obs(features)):
            predictions.append(self._predict_sample(features[obs,:],self.root))
        return np.array(predictions)