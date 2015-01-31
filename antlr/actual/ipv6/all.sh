rm -rf results
mkdir results

echo "Running nogoto"
./nogoto.sh > results/nogoto_out
#./nogoto.sh

echo "Running goto"
./goto.sh > results/goto_out
#./goto.sh

echo "Running handopt"
./handopt.sh > results/handopt_out
#./handopt.sh
