

#### Prompt design for SensorCaps generation. We prompt GPT-4o-mini with specific instructions to generate a caption for the sensor data in 20-25 words and generate a step-by-step narration of the activity's sensor reading.



<table><thead>
  <tr>
    <th colspan="2">Instruction for Generating SensorCaps</th>
  </tr></thead>
<tbody>
  <tr>
      <td colspan="2">Please consider yourself to be an expert on gyroscope and accelerometer sensor information given as metadata of IMU datasets. You are given the IMU sensor readings of human activity. The user also provides a brief summary of the event followed by 'Summary:'. They also give you gyroscopic and accelerometer sensor data followed by 'Gyroscope:' and 'Accelerometer:' respectively. They are written in a Python list of lists format and contain x, y, and z-axis data, respectively. Please pay attention to the values and the signs. Additional information computed for each axis is provided afterward, with the windows for moving averages being 12. You should provide comprehensive detail of at least a couple of characteristic IMU features for that event within 20-25 words, followed by 'Features:.' The IMU features should be concise and descriptive. Separate multiple features with commas. Derive the list of features based on the given sensor data. Then, narrate the temporal event with details that are context-aware based on the sensor data, followed by 'Narration:' in a step-by-step fashion, analyzing it within 500 words or less. Please analyze even the small movements thoroughly in a logical and contextual manner, utilizing a deductive thought process while being aware of the knowledge regarding the meaning of the sensor data. Use descriptive terms that aren't vague.</td>
  </tr>
  <tr>
    <td>Prompt Input Structure</td>
    <td>Output Structure</td>
  </tr>
  <tr>
    <td>1. Activity class label <br> 2. Gyroscope sensor data <br>  3. Accelerometer sensor data <br>  4. Sensor location <br>  5. Handcrafted features: moving averages, FFT, min, max, median, variance for each axis</td>
    <td>1. Characteristics IMU features in natural language <br> 2. Narration of the activity's sensor readings</td>
  </tr>
</tbody></table>
