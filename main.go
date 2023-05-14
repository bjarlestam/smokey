package main

import (
	"context"
	"encoding/gob"
	"flag"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"cloud.google.com/go/bigquery"
	"google.golang.org/api/iterator"
)

type Context struct {
	UserID    string
	TimeOfDay string
	Weekday   string
	Device    string
}

type Bandit struct {
	ItemID         string
	ContextRewards map[Context]float64
}

func (b *Bandit) Pull(ctx Context) float64 {

	reward, ok := b.ContextRewards[ctx]
	if !ok {
		return 0.0 // default to 0 reward if we dont have any data for the given context
	}
	return reward
}

type Strategy interface {
	SelectBandit(ctx Context) *Bandit
	UpdateReward(ctx Context, b *Bandit, reward float64)
}

type EpsilonGreedyStrategy struct {
	Epsilon float64
	Bandits []*Bandit
	Rewards map[Context][]float64
	Counts  map[Context][]int
}

type TrainingData struct {
	UserID    string                `bigquery:"user_id"`
	ItemID    string                `bigquery:"item_id"`
	Timestamp bigquery.NullDateTime `bigquery:"impression_time"`
	HasClick  bool                  `bigquery:"was_clicked"`
	Device    string                `bigquery:"device"`
}

func (s *EpsilonGreedyStrategy) SelectBandit(ctx Context) *Bandit {
	if rand.Float64() < s.Epsilon || len(s.Rewards[ctx]) == 0 {
		// Explore
		return s.Bandits[rand.Intn(len(s.Bandits))]
	}

	// Exploit
	maxReward := s.Rewards[ctx][0]
	maxIndex := 0
	for i, reward := range s.Rewards[ctx] {
		if reward > maxReward {
			maxReward = reward
			maxIndex = i
		}
	}

	return s.Bandits[maxIndex]
}

func (s *EpsilonGreedyStrategy) UpdateReward(ctx Context, b *Bandit, reward float64) {
	for i := range s.Bandits {
		if s.Bandits[i] == b {
			s.Counts[ctx][i]++
			s.Rewards[ctx][i] = ((s.Rewards[ctx][i] * float64(s.Counts[ctx][i]-1)) + reward) / float64(s.Counts[ctx][i])
		}
	}
}

func (s *EpsilonGreedyStrategy) SaveState(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := gob.NewEncoder(file)
	err = encoder.Encode(s)
	if err != nil {
		return err
	}

	return nil
}

func (s *EpsilonGreedyStrategy) LoadState(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(s)
	if err != nil {
		return err
	}

	return nil
}

func trainModel() {
	contexts, bandits := getTrainingData()

	strategy := &EpsilonGreedyStrategy{
		Epsilon: 0.1, // fraction of exploration 0.1 = 10% exploration
		Bandits: bandits,
		Rewards: make(map[Context][]float64),
		Counts:  make(map[Context][]int),
	}

	log.Print("Training...")

	// Train the model
	for _, ctx := range contexts {
		strategy.Rewards[ctx] = make([]float64, len(bandits))
		strategy.Counts[ctx] = make([]int, len(bandits)) // initialize counts to zero
		for i := 0; i < 10000; i++ {
			bandit := strategy.SelectBandit(ctx)
			reward := bandit.Pull(ctx)
			strategy.UpdateReward(ctx, bandit, reward)
		}
	}

	// Save the state
	filename := "strategy.gob"
	log.Printf("Saving modeld as %s", filename)
	err := strategy.SaveState(filename)
	if err != nil {
		log.Fatal(err)
	}
}

func loadModelAndSelectAnItem(userId *string, timeOfDay *string, weekday *string, device *string) {

	strategy := &EpsilonGreedyStrategy{
		Epsilon: 0.1,
		Bandits: nil,
		Rewards: make(map[Context][]float64),
		Counts:  make(map[Context][]int),
	}
	log.Print("Loading model")
	filename := "strategy.gob"
	strategy.LoadState(filename)

	log.Print("Selecting an item to recommend")
	// define your context
	ctx := Context{UserID: *userId, TimeOfDay: *timeOfDay, Weekday: *weekday, Device: *device}
	// strategy selects a bandit based on the context
	bandit := strategy.SelectBandit(ctx)

	log.Printf("Recommend item: %s\n", bandit.ItemID)
}

func getTrainingData() ([]Context, []*Bandit) {
	ctx := context.Background()

	// Create a client.
	client, err := bigquery.NewClient(ctx, "<bigquery project>")
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	q := client.Query(`
		SELECT 
		user_id,
		item_id,
		impression_time,
		was_clicked,
		device
		FROM <dataset>
	`)
	it, err := q.Read(ctx)
	if err != nil {
		log.Fatalf("Failed to initiate reading: %v", err)
	}

	// Create empty contexts and bandits.
	contexts := []Context{}
	bandits := []*Bandit{}

	for {
		var row TrainingData
		err := it.Next(&row)
		if err == iterator.Done {
			break
		}
		if err != nil {
			log.Fatalf("Failed to read data: %v", err)
		}

		// Determine time of day and day of week.
		var timeOfDay, weekday string
		if row.Timestamp.Valid {
			hour := row.Timestamp.DateTime.Time.Hour
			if hour < 4 {
				timeOfDay = "night"
			} else if hour < 12 {
				timeOfDay = "morning"
			} else if hour < 18 {
				timeOfDay = "afternoon"
			} else if hour < 22 {
				timeOfDay = "evening"
			} else {
				timeOfDay = "night"
			}

			// Convert civil.DateTime to time.Time to get the weekday.
			t := time.Date(row.Timestamp.DateTime.Date.Year, row.Timestamp.DateTime.Date.Month, row.Timestamp.DateTime.Date.Day, 0, 0, 0, 0, time.UTC)
			weekday = strings.ToLower(t.Weekday().String())
		}

		// Create a new context.
		ctx := Context{row.UserID, timeOfDay, weekday, row.Device}
		contexts = append(contexts, ctx)

		// Check if the item already exists in bandits.
		found := false
		for _, bandit := range bandits {
			if bandit.ItemID == row.ItemID {
				// The item exists, update the context rewards.
				if row.HasClick {
					bandit.ContextRewards[ctx] += 1.0
				} else {
					bandit.ContextRewards[ctx] -= 0.1 // subtract a small penalty for not getting a click
				}
				found = true
				break
			}
		}
		if !found {
			// The item does not exist, create a new bandit.
			reward := 0.0
			if row.HasClick {
				reward = 1.0
			}
			bandit := &Bandit{
				ItemID: row.ItemID,
				ContextRewards: map[Context]float64{
					ctx: reward,
				},
			}
			bandits = append(bandits, bandit)
		}
	}

	log.Printf("Fetched %d rows of training data", it.TotalRows)
	log.Printf("There are %d bandits to choose from", len(bandits))

	return contexts, bandits
}

func main() {

	// handle command line options
	train := flag.Bool("train", false, "Train the model")
	userId := flag.String("user", "", "User ID")
	timeOfDay := flag.String("time", "", "Time of day [morning|afternoon|evening|night]")
	weekday := flag.String("weekday", "", "Weekday")
	device := flag.String("device", "", "Device")
	flag.Parse()

	// If the train flag is present, train the model; otherwise, load the model and make selection
	if *train {
		trainModel()
	} else {
		loadModelAndSelectAnItem(userId, timeOfDay, weekday, device)
	}
}
