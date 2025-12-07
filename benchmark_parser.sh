

#!/bin/bash  

#benchmark_parser.sh is takes the output from benchmark.py
#and uses awk and sed to filter out the numbers for each metric, for use in the spreadsheet

#to use it, call benchmark.py and use > to put the inputs into a file like "ddqn_1.txt"
#then change the variables here to match that, then an output fiel
INPUT_FILE="/benchmark_outputs/ddqn_1.txt"
OUTPUT_FILE="benchmark_outputs/ddqn_1_parsed.txt"

echo "taking from ${INPUT_FILE}, into ${OUTPUT_FILE}"


#with extra time, would have used a loop, and structured outputs like csv, but this was fast enough for only 10 runs

echo "Test Density" >> $OUTPUT_FILE
grep "density=" $INPUT_FILE | awk -F '=' '{print $2}' | awk '{print $1}' | sed 's/,//g' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE 


echo "Duration" >> $OUTPUT_FILE
grep "duration=" $INPUT_FILE | awk -F ', ' '{print $2}' | awk -F '=' '{print $2}' | sed 's/s//; s/,//g' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE 

echo "Avg Reward" >> $OUTPUT_FILE
grep "Avg Reward" $INPUT_FILE | sed 's/.*: *//' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "Collision Rate" >> $OUTPUT_FILE
grep "Collision Rate" $INPUT_FILE | sed 's/.*: *//' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "Avg Speed" >> $OUTPUT_FILE
grep "Avg Speed" $INPUT_FILE | sed 's/.*: *//' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "Avg Distance" >> $OUTPUT_FILE
grep "Avg Distance" $INPUT_FILE | sed 's/.*: *//' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "RMS Accel" >> $OUTPUT_FILE
grep "RMS Accel" $INPUT_FILE | sed 's/.*: *//' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "RMS Jerk" >> $OUTPUT_FILE
grep "RMS Jerk" $INPUT_FILE | sed 's/.*: *//' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "Action Distribution" >> $OUTPUT_FILE
grep "Action distribution" $INPUT_FILE | awk '{print $NF-2}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "LANE_LEFT %" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "LANE_LEFT" | awk -F '[%()]' '{print $1}' | awk '{print $NF}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "LANE_LEFT #" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "LANE_LEFT" | awk -F '[()]' '{print $2}' | awk '{print $1}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "IDLE %" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "IDLE" | awk -F '[%()]' '{print $1}' | awk '{print $NF}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "IDLE #" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "IDLE" | awk -F '[()]' '{print $2}' | awk '{print $1}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "LANE_RIGHT %" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "LANE_RIGHT" | awk -F '[%()]' '{print $1}' | awk '{print $NF}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "LANE_RIGHT #" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "LANE_RIGHT" | awk -F '[()]' '{print $2}' | awk '{print $1}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "FASTER %" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "FASTER" | awk -F '[%()]' '{print $1}' | awk '{print $NF}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "FASTER #" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "FASTER" | awk -F '[()]' '{print $2}' | awk '{print $1}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "SLOWER %" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "SLOWER" | awk -F '[%()]' '{print $1}' | awk '{print $NF}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "SLOWER #" >> $OUTPUT_FILE
grep -A 5 "Action distribution" $INPUT_FILE | grep "SLOWER" | awk -F '[()]' '{print $2}' | awk '{print $1}' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

