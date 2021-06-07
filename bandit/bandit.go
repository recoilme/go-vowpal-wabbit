package bandit

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"

	wabbit "github.com/recoilme/go-vowpal-wabbit"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type Bandit struct {
	sync.Mutex
	vw *wabbit.VW
}

type Cntxt struct {
	Prop map[string]string
}

type Action struct {
	ID   string
	Prop map[string]string
	Prob float32
}

func newBandit(options string) (*Bandit, error) {
	if options == "" {
		options = "--cb_explore_adf -b 18 -q UA --quiet --epsilon 0.2"
	}
	vw, err := wabbit.New(options)
	if err != nil {
		return nil, err
	}
	return &Bandit{vw: vw}, err
}

func (c *Cntxt) Properties() string {
	return mapstr(c.Prop)
}

func (a *Action) Properties() string {
	return mapstr(a.Prop)
}

func mapstr(m map[string]string) string {
	if len(m) == 0 {
		return ""
	}
	sb := strings.Builder{}
	for k, v := range m {
		sb.WriteString(k)
		sb.WriteString("=")
		sb.WriteString(v)
		sb.WriteString(" ")
	}
	return strings.TrimSuffix(sb.String(), " ")
}

/*
shared |Context Context=Tom time_of_day=morning
|Action article=politics
|Action article=sports
|Action article=music
|Action article=food
*/
func (b *Bandit) PredictLines(c *Cntxt, act []*Action) []string {
	lines := make([]string, 0, len(act)+1)
	ctxt := fmt.Sprintf("shared |User %s", c.Properties())
	lines = append(lines, ctxt)
	for _, a := range act {
		line := fmt.Sprintf("|Action %s", a.Properties())
		lines = append(lines, line)
	}

	return lines
}

// Predict return Actions ordered by probability
/*
[shared |User usr=tom time=aft |Action id=politics |Action id=sports |Action id=music |Action id=food |Action id=finance |Action id=health |Action id=camping]
*/
func (b *Bandit) Predict(c *Cntxt, act []*Action) ([]*Action, error) {
	b.Lock()
	defer b.Unlock()
	lines := b.PredictLines(c, act)
	//fmt.Println(lines)
	examples, err := b.getExamples(lines)
	if err != nil {
		return nil, err
	}
	err = b.vw.MultiLinePredict(examples)
	defer examples.Finish() //free
	if err != nil {
		return nil, err
	}

	// получение вероятностей для examples

	//fmt.Println( examples[0].GetActionScores(), examples[0].GetActions())
	scores := examples[0].GetActionScores()
	for i, ind := range examples[0].GetActions() {
		act[ind].Prob = scores[i]
	}

	// сортируем по вероятностям
	sort.Slice(act, func(i, j int) bool {
		return act[i].Prob > act[j].Prob
	})

	return act, nil
}

func (b *Bandit) getExamples(list []string) (wabbit.ExampleList, error) {
	examples := make([]*wabbit.Example, 0)
	for _, v := range list {
		ex, err := b.vw.ReadExample(v)
		if err != nil {
			return nil, err
		}
		examples = append(examples, ex)
	}
	return examples, nil
}

func (b *Bandit) RewardLines(c *Cntxt, act []*Action, id int, reward float32) []string {
	lines := make([]string, 0, len(act)+1)
	if id >= len(act) {
		return lines
	}
	ctxt := fmt.Sprintf("shared |User %s", c.Properties())
	lines = append(lines, ctxt)
	for i, a := range act {
		line := fmt.Sprintf("|Action %s", a.Properties())
		//fmt.Println("a.id", a.ID, id, a.Prop["Type"], i, a)
		if i == id {
			line = fmt.Sprintf("%d:%f:%f %s", 0, -1*reward, a.Prob, line)
		}
		lines = append(lines, line)
	}

	return lines
}

// Learn, take reward (not cost)
// Use cost for penalti
// Winned action must be on zero place
func (b *Bandit) Reward(c *Cntxt, act []*Action, id int, reward float32) error {
	b.Lock()
	defer b.Unlock()
	lines := b.RewardLines(c, act, id, reward)
	if len(lines) == 0 {
		return errors.New("id more act len")
	}
	//fmt.Println("reward", id, lines)
	examples, err := b.getExamples(lines)
	defer examples.Finish()
	if err != nil {
		return err
	}
	return b.vw.MultiLineLearn(examples)
}

// Probability mass function. Return winned Action on first place
// So, given a list [0.7, 0.1, 0.1, 0.1], we would choose the first item with a 70% chance
// Place on zero index wined action
func Select(act []*Action) int {
	sum := float32(0)
	for _, a := range act {
		sum += a.Prob
	}
	r := rand.Float32() * sum
	sum = 0.
	for i, a := range act {
		sum += a.Prob
		if sum >= r {
			if i == 0 {
				return i
			}
			//act[0], act[i] = act[i], act[0] //swap
			//act[0].Prob = act[1].Prob+0.01
			return i
		}
	}
	return -1
}

func serializeToJSON(m map[string]string) string {
	b, err := json.Marshal(m)
	if err != nil {
		panic(err)
	}

	return string(b)
}
