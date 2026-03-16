#!/bin/bash
  
ulimit -n 100000
APP=$@
local_rank=$((${MPI_LOCALRANKID}%8))

case ${local_rank} in
[0])
        numa_node="0"
        device_id="0"
        network_id="shca_0:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
[1])
        numa_node="3"
        device_id="1"
        network_id="shca_1:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
[2])
        numa_node="2"
        device_id="2"
        network_id="shca_0:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
[3])
        numa_node="1"
        device_id="3"
        network_id="shca_1:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
[4])
        numa_node="4"
        device_id="4"
        network_id="shca_2:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
[5])
        numa_node="7"
        device_id="5"
        network_id="shca_3:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
[6])
        numa_node="6"
        device_id="6"
        network_id="shca_2:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
[7])
        numa_node="5"
        device_id="7"
        network_id="shca_3:1"

        #export HIP_VISIBLE_DEVICES=${device_id}
        export FI_UCX_DEVICES=${network_id}

        numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
        ;;
esac
