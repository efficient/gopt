rm -rf results
mkdir results

echo "Running nogoto"
./nogoto.sh > results/nogoto_out

echo "Running goto"
./goto.sh > results/goto_out

echo "Running handopt"
./handopt.sh > results/handopt_out
