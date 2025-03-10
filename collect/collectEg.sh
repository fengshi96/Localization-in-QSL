# Create or clear energies.dat file at the beginning
> energies.dat

for hin in $(seq 0.00 0.01 0.16)
do
    # Extract the last occurrence of the line containing "energy" from the file
    energy_line=$(grep "energy" ../hin_$hin/gs.out | tail -n 1)

    # Extract the real part of the energy (handle both + and - signs in the complex part)
    real_energy=$(echo $energy_line | sed -n 's/.*energy of pure model: (\([0-9.-]*\)[+-].*/\1/p')
    
    # Append the hin value and the real energy to energies.dat
    echo "$hin $real_energy" >> energies.dat
done
