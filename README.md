# Smokey
A minimal implementation of a Contextual Bandit for selecting an item to recommend for a specific user. It uses an epsilon-greedy-strategy to alternate exploration and exploitation. The amount of exploration can be tweaked with the Epsilon parameter. Currently it is configured to do 10% exploration.

## Training the model
The model can be trained with the following command:
```
go run main.go --train
```
then training data is fetched from a big query table (you need to modify the `<bigquery project>` and the `<dataset>`), 
The dataset should include the following columns
* user_id,
* item_id,
* impression_time,
* was_clicked,
* device

The model is trained on that data and then saved to the file `strategy.gob`

## Using the model to select an item to recommend
The model can select a good item given a specific context. The context consists of:
* user
* time [morning|afternoon|evening|night]
* weekday [monday|tuesday|wednesday|thursday|friday|saturday|sunday]
* device [mobile|desktop|tablet|tv]
```
go run main.go --user 434521 --time morning --weekday monday --device mobile
```
