
> results.txt
for nreact in 10 100 500 1000 2500 5000
do
    for nspecies in 10 100 500 1000 1500 2000
    do
        echo "reactions:" ${nreact} "; species:" ${nspecies} >> results.txt
        cmake -DNREACTIONS=${nreact}  -DNSPECIES=${nspecies} ..
        make
        ./test/derivative/derivative_test >> results.txt
    done
done
