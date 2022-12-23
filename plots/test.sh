
# run from within the build directory

# requries https://github.com/nico/ninjatracing
# the results can be viewed in chrome https://crascit.com/2022/06/24/build-performance-insights/

> results.txt
mkdir data
cd ../build
for nreact in 10 100 500 1000 2500 5000
do
    for nspecies in 10 100 500 1000 1500 2000
    do
        > .ninja_log
        echo "reactions:" ${nreact} "; species:" ${nspecies} >> results.txt
        cmake -DCMAKE_C_FLAGS=-ftime-trace \
              -DCMAKE_CXX_FLAGS=-ftime-trace \
              -DNREACTIONS=${nreact} \
              -DNSPECIES=${nspecies} \
              -DCMAKE_BUILD_TYPE=Release \
              -G Ninja ..
        ninja
        python ninjatracing .ninja_log > ../plots/data/cmake_build_trace_nreact_${nreact}_nspecies_${nspecies}.json
        ./test/derivative/derivative_test >> ../plots/results.txt
    done
done
