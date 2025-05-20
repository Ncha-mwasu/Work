import sched, time

class SleepScheduler():
    def __init__(self, cluster_nodes):
        self.cluster_nodes = cluster_nodes
        self.len_of_cluster_nodes = len(cluster_nodes)
        # self.queue = []
        # self.next_active =
        self.next_node_to_transmit_index = 0
        for node in self.cluster_nodes:
            node.scheduler = self

       
        

    # def schedule(self):
    #     self.

    def get_next_to_transmit(self):
        return self.cluster_nodes[self.next_node_to_transmit_index]

    def handler(self):
        #print("handled")
        if self.next_node_to_transmit_index < self.len_of_cluster_nodes-1:
            self.next_node_to_transmit_index += 1
        else:
            self.next_node_to_transmit_index = 0


#wn = turtle.Screen()
# turtle.ontimer(handler, 3000)


