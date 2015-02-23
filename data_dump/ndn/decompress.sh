# Decompress the fib_1010.lzma file without deleting it
cp fib_1010.lzma temp
lzma -d fib_1010.lzma
mv temp fib_1010.lzma
