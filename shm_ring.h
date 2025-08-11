#ifndef SHM_RING_H
#define SHM_RING_H

/*
 * Simple shared memory ring buffer used to exchange variable sized
 * messages between a single producer and single consumer.  The layout
 * matches the implementation used by scripts/shm_ring.py.
 *
 *  +--------------+---------------------------------------------------+
 *  | field        | description                                      |
 *  +--------------+---------------------------------------------------+
 *  | magic        | constant 0x52494E47 ('RING')                      |
 *  | size         | size of data region in bytes                      |
 *  | head         | read offset                                       |
 *  | tail         | write offset                                      |
 *  | data[size]   | ring storage                                      |
 *  +--------------+---------------------------------------------------+
 *
 *  Each message is stored as:
 *       uint32_t length  - number of bytes following (type+payload)
 *       uint8_t  type    - message type (0 trade, 1 metric)
 *       uint8_t  payload[length-1]
 *
 *  A zero length message is used as a wrap marker when the producer
 *  reaches the end of the buffer.
 */

#include <stdint.h>
#include <string.h>

#define SHM_RING_MAGIC 0x52494E47u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t magic;
    uint32_t size;
    volatile uint32_t head;
    volatile uint32_t tail;
    uint8_t  data[1]; /* flexible array member */
} shm_ring_t;

#define SHM_RING_HEADER (sizeof(shm_ring_t) - 1)

static inline size_t shm_ring_buffer_size(size_t capacity) {
    return SHM_RING_HEADER + capacity;
}

static inline uint32_t shm_ring_space(const shm_ring_t *r) {
    uint32_t head = r->head, tail = r->tail, size = r->size;
    if (tail >= head)
        return size - (tail - head) - 1;
    return head - tail - 1;
}

static inline int shm_ring_write(shm_ring_t *r, uint8_t type,
                                 const void *data, uint32_t len) {
    uint32_t payload = len + 1; /* include type */
    uint32_t needed = 4 + payload;
    if (needed > r->size)
        return -1; /* message too large */
    if (shm_ring_space(r) < needed)
        return 0; /* insufficient space */

    uint32_t tail = r->tail;
    if (tail + needed > r->size) {
        /* mark wrap and start from beginning */
        *(uint32_t *)(r->data + tail) = 0;
        tail = 0;
    }
    *(uint32_t *)(r->data + tail) = payload;
    tail += 4;
    r->data[tail] = type;
    memcpy(r->data + tail + 1, data, len);
    tail = (tail + payload) % r->size;
    r->tail = tail;
    return 1;
}

static inline int shm_ring_read(shm_ring_t *r, uint8_t *type,
                                void *out, uint32_t *len) {
    uint32_t head = r->head;
    uint32_t tail = r->tail;
    uint32_t size = r->size;
    if (head == tail)
        return 0; /* empty */
    if (head + 4 > size)
        head = 0;
    uint32_t payload = *(uint32_t *)(r->data + head);
    if (payload == 0) {
        head = 0;
        payload = *(uint32_t *)(r->data + head);
    }
    uint32_t start = head + 4;
    if (payload < 1)
        return -1; /* corrupt */
    if (type)
        *type = r->data[start];
    if (out && len && *len >= payload - 1)
        memcpy(out, r->data + start + 1, payload - 1);
    if (len)
        *len = payload - 1;
    head = (start + payload) % size;
    r->head = head;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* SHM_RING_H */
