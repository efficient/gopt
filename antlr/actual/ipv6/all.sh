rm -rf results
mkdir results

echo "Running nogoto"
./nogoto.sh

echo "Running goto"
./goto.sh

echo "Running handopt"
./handopt.sh
