from transformers import pipeline

summarizer = pipeline(
    "summarization", model="dhrumeen/mt5-small_summarization")

result = summarizer("The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.")

a = 'Finance Minister Nirmala Sitharaman termed the Interim Budget 2024-25 “secularism” in action. “People on the ground are talking about programmes, this is secularism in action, ” she said.\
In an exclusive interview with Managing Director of Network18 Rahul Joshi, Sitharaman said the Centre has not shown any difference between any community. “Projects reach everyone who deserves the benefits,” she said while adding that more has to be done to ensure the Centre works with the states.According to her, the power of word of mouth is “very strong”. “When a beneficiary on the ground truly gets a benefit, they believe what the government says,” said the Finance Minister. “This government is serving the common people in letter and spirit and that is recognised by the people themselves.”\
She said while speaking on international rating agencies “I would think they do their job, but periodically it’s our business also to bring it to their notice that the economy, particularly emerging market economy like India despite the odds we are doing a lot of systemic reforms which are bearing the results now.”\
Sitharaman on Thursday presented her sixth consecutive budget. In less than an hour-long budget speech, she presented the Modi government’s achievements in the last 10 years that transformed India from being a ‘fragile’ economy to the world’s fastest-growing major economy.Impact of digitisation on rural economy yet to be measured: FM Sitharaman to Network18 The impacts of digitisation and connectivity on the rural economy are not coming through in current economic indicators, FM Nirmala Sitharaman said in an interview with Network 18’s Editor-in-Chief Rahul Joshi on February 2.\
“Indicators with which we are looking at the rural economy may vary, newer indicators may tell us a different story. FMCG data is one indicator. But better connectivity, digitisation (in rural areas) are yet to be measured,” said the Finance Minister a day after she presented the interim Budget.\
She also said that college recruitments, and hiring in top institutions like IIMs are important, but jobs created in middle and lower order are not being counted when talking about employment in the economy.\
“I am not sure I will be able to describe what is happening in the rural economy. Let us recognise there is a lot of shift in employment, let us recognise migration is now looking at redefining itself,” she further said.\
Sitharaman on Thursday presented her sixth consecutive budget with a speech lasting 56 minutes, her shortest-ever.\
She presented the Modi government’s achievements in the last 10 years that transformed India from being a ‘fragile’ economy to the world’s fastest-growing major economy.\
Sitharaman also holds the distinction of delivering the longest budget speech at 2.40 hours in 2020.\
In 2019, as India’s first full-time woman finance minister, Sitharaman’s budget speech had lasted for two hours and 17 minutes. In 2021, her speech lasted for one hour and 50 minutes, followed by 92 minutes in 2022 and 87 minutes in 2023.'

result = summarizer(a, max_length=100, min_length=30, do_sample=format)
print(result)
