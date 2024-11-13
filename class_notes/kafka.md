# Kafka

A topic partition is the unit of parallelism in Kafka
### Producers, Brokers and Consumers
Producers and Brokers:
- Writes to different partitions can be done in parallel
- Parallelism frees up hardware resources for operations like compression

Consumers:
- Grouped together into several consumer groups
    - Each consumer takes a full Copy of the data **EXAM**
    - Each group has it's own access to the contents of the topics. But ONLY one consumer in each group per partition
    - More partitions you have in a kafka cluster leads to higher system in that case

Kafka ensures **messages with identical keys are consistently routed to the same partition** preserving the integrity of order-dependent applications

