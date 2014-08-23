echo ""
echo "Nogoto:"
sudo numactl --physcpubind 0 --interleave 0 ./nogoto

echo ""
echo "Handopt:"
sudo numactl --physcpubind 0 --interleave 0 ./handopt

echo ""
echo "Goto:"
sudo numactl --physcpubind 0 --interleave 0 ./goto
