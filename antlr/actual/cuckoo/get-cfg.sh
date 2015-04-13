# Remove all old ps files from /tmp
rm /tmp/*.ps

# Create the bitcode file
clang -emit-llvm -c simple.c

# Get the CFG from the bitcode file
opt --view-cfg-only simple.bc

# Copy the generated ps file here
cp /tmp/*.ps .
