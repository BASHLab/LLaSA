

### Prompt design for sensor-aware QA evaluation. GPT-4o is prompted with specific instruction to evaluate the quality of the answer generated by LLaSA, GPT-3.5-Turbo and GPT-4o-mini using a quality score and brief summary of the assessement.



---


<table><thead>
  <tr>
    <th colspan="2">Instruction for Assessing Sensor-aware Multimodal LLMs</th>
  </tr></thead>
<tbody>
  <tr>
      <td colspan="2">Assume that you are an expert at assessing the quality of predicted answers based on questions related to IMU data and given information. Please use the ’Standard Answer,’ ’Activity Label,’ and ’Sensor Location’ to assess the ’Predicted Answer’ based on the correctness, completeness, consistency, and helpfulness of the predicted answer. Every question is given accelerometer and gyroscope data. So, do not only look at the general answer; check if the answer provided any insights from the data. Provide a single overall score for each answer on a scale of 0 to 100 using the following format: <br><em>Quality score:</em> <br>Then, give a brief summary of the assessment in 1-2 sentences, followed by "Assessment:". Be specific in evaluating the accuracy, detail, and relevance of the predicted answer in relation to the expected standard.</td>
  </tr>
  <tr>
    <td>Prompt Input Structure</td>
    <td>Output Structure</td>
  </tr>
  <tr>
    <td>1. Standard Answer <br>2. Activity Label <br>3. Sensor Location <br>4. Predicted Answer </td>
    <td>1. Quality score (0-100) <br>2. Assessment Summary: brief evaluation of answers' <br>quality based on correctness, completeness, <br>consistency, and helpfulness</td>
  </tr>
</tbody></table>
