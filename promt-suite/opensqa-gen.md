

#### Prompt design to generate OpenSQA. GPT-4o-mini is prompted with specific instructions and necessary input to generate high-quality five question-answers pair for LLMs.



<table><thead>
  <tr>
    <th colspan="2">Instruction for Generating OpenSQA</th>
  </tr></thead>
<tbody>
  <tr>
      <td colspan="2">Please consider yourself an expert on gyroscope and accelerometer sensor information for IMU datasets. You are given the IMU sensor readings of human activity. The user also provides a brief summary of the event followed by 'Summary:' They also give you gyroscopic and accelerometer sensor data followed by 'Gyroscope:' and 'Accelerometer:' respectively. Characteristic IMU features of the event is written after 'Features:' Finally, there's a temporal narration of the events after 'Narration:' Please generate five detailed question-answer pairs that require step-by-step logical deductive thinking and knowledgeable analysis of the sensor data. Please make the questions complex but logical so that they require information that can be derived based on the vast knowledge of sensor data and  IMU activities. The questions and answers need to acknowledge the context given by the user. Please write them in the following list format: 1. Q: Question  A: Answer</td>
  </tr>
  <tr>
    <td>Prompt Input Structure</td>
    <td>Output Structure</td>
  </tr>
  <tr>
    <td>1. Activity class label <br> 2. Gyroscope sensor data <br> 3. Accelerometer sensor data <br> 4. Sensor Location <br> 5. Characteristic IMU features (SensorCaps) <br> 6. Temporal narration of the activity (SensorCaps)</td>
    <td>Five pairs of questions and answers.</td>
  </tr>
</tbody></table>
