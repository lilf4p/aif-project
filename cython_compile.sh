# Compile Cython files (.pyx) into a complete (local) Python extension.
# This is required to use Cython version of crossover and mutation operators for testing.
# Solution for multi-OS is taken from: https://stackoverflow.com/questions/394230/how-to-detect-the-os-from-a-bash-script
case "$OSTYPE" in
  linux*)
    echo "Using a Linux operating system"
    python3 setup.py build_ext --inplace
    ;;
  msys*)
    echo "Using a Windows operating system"
    python setup.py build_ext --inplace
    ;;
  cygwin*)
    echo "Using CygWin for Windows"
    python setup.py build_ext --inplace
    ;;
  darwin*)
    echo "Using a MAC operating system"  # todo check that this actually works for MAC!
    python setup.py build_ext --inplace
    ;;
  *)
    echo "Using another OS: " $OS
esac

read -p "Press Enter to finish"