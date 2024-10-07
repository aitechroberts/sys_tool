# Notes on Apache Kafka
 Developed at LinkedIn in 2011

Kafka is a messaging-based log aggregator

## Key Concepts
Topic
 - Stream of messages of a particular type
 - A producer can publish messages to a topic to a broker
Brokers
 - Set of servers storing the published messages
 - A consumer can subscribe to one or more topics from the brokers
 - Consumes the subscribed messages by PULLING from the brokers (as opposed to a push model)

#### Sample Producer Code
'''
producer = new Producer(…);
message = new Message(“test message str”.getBytes());
set = new MessageSet(message);
producer.send(“topic1”, set);
'''

#### Sample consumer code:
'''
streams[] = Consumer.createMessageStreams(“topic1”, 1)
for (message : streams[0]) {
bytes = message.payload();
// do something with the bytes
}'''

## Process
To subscrib to a topic, a consumer first creates one or more message streams for the topic. The messages published to that topic will be evely distributed into these sub-streams. {Because this is a stream and must be discretized] Each message stream provides an iterator interface over the continual stream of messages being produced. The consumer then iterates over every message in the stream and processes the payload of the message. Message stream iterator never terminates. If there is are no messages to consume, the iterator blocks until new messages are published.

Point-to-Point
- multiple consumers jointly consume a single copy of all messages in a topic

Publish/Subscribe (Pub/Sub)
- multiple consumers retrieve its own copy of a topic

## Architecture
To balance load on a Kafka cluster consisting of multiple brokers, a topic is divided into multiple partitions and each stores one or more of those partitions.

### Single Partition
##### Simple Storage
- Each partition of a topic corresponds to a logical log
- Physically, a log is implemented as a set of segment files of approx. same size
- When a producer publishes a message to a partition, broker appends message to the last segment file.
- For better performance, segment files are flushed to disk only after a configurable number of messages has been published OR a certain time period has elapsed. A message is only exposed to consumers after it is flushed

There is no message ID in Kafka, rather, it is addressed by its logical offset in the log. Offset being the bits/bytes memory offset of the message to compute the logical distance on disk/memory of the next message.

##### Consumption

A consumer always consumes messages from a particular partition sequentially. By doing so, it acknowledges the message offset location in memory implying that the consumer has received all messages prior to that offset in the segment file.

Under it all, the consumer is issuing asynchronous pull requests to the broker to have a buffer of data ready for the application to consume. Each pull request contains the offset of the message from which the request begins and an acceptable number of bytes to fetch.

Each broker keeps a sorted list of offsets indicating where in each segment file the last message was sent as well as that segment file's first message's offset. The broker locates the segment file where the requested message resides by searching its sorted offset list, and satisfies the rquest by sending data back to the consumer. After a consumer receives a message, it uses that message's offset to start next pull request. 

##### Efficient Transfer

Although the end consumer API iterates one message at a time, each pull request retrieves multiple messages up to a certain size.

Another unconventional choice that we made is to avoid explicitly caching messages in memory at the Kafka layer. Instead, we rely on the underlying file system page cache. This has the main benefit of avoiding double buffering---messages are only cached in the page cache. This has the additional benefit of retaining warm cache even when a broker process is restarted. Since Kafka doesn’t cache messages in process at all, it has very little overhead in garbage collecting its memory, making efficient implementation in a VM-based language feasible.

Since both the producer and consumer access the segment files sequentially, with the consumer often lagging the producer by a small amount, normal operating system caching heuristics are very effective (specifically write-through caching and readahead). We have found that both the production and the consumption have consistent performance linear to the data size, up to many terabytes of data.

In addition we optimize the network access for consumers. Kafka is a multi-subscriber system and a single message may be consumed multiple times by different consumer applications.

#### Stateless Broker

Since the consumer maintains how much it has consumed rather than the Kafka controlled broker, Kafka is a stateless broker. But this makes it tricky to delete a message when Kafka doesn't know whether all subscribers have consumed it. Kafka solves the majority of this problem with a time-based retention policy, automatically deleting a message at the end of that time. 

Side benefit of this is that consumers can deliberately rewind back to an old offset and re-consume data. This violates the common contract of a queue, but proves to be an essential feature for many consumers. For example, when there is an error in application logic in the consumer, the application can re-play certain messages after the error is fixed This is particularly important to ETL data loads into our data warehouse or Hadoop system.

As another example, the consumed data may
be flushed to a persistent store only periodically (e.g, a full-text indexer). If the consumer crashes, the unflushed data is lost. In this case, the consumer can checkpoint the smallest offset of the unflushed messages and re-consume from that offset when it’s restarted. We note that rewinding a consumer is much easier to support in the pull model than the push model.

**Check out Kafka Mirror Maker and Brooklin**
- LinkedIn used Kafka Mirror Maker to copy data from the local to the aggregate cluster. Later, they encountered a scaling issue with this replication tool, so they switched to Brooklin, an internal solution that allows data to be streamed across different data stores.
[link here](https://blog.det.life/how-did-linkedin-handle-7-trillion-messages-daily-with-apache-kafka-07a167f1a949)
[LI_Engineering](https://engineering.linkedin.com/kafka/running-kafka-scale)
