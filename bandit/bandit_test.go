package bandit

import (
	"fmt"
	"log"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Ctxt(t *testing.T) {
	c := &Cntxt{Prop: map[string]string{
		"age": "21",
	}}
	assert.Equal(t, "age=21", c.Properties())
}

func Test_Pred(t *testing.T) {
	act1 := &Action{
		ID: "1",
		Prop: map[string]string{
			"theme": "sport",
		}}
	assert.Equal(t, "theme=sport", act1.Properties())

	usr := &Cntxt{Prop: map[string]string{
		"sex": "f",
	}}

	act2 := &Action{
		ID: "2",
		Prop: map[string]string{
			"theme": "music",
		}}

	bnd, err := newBandit("--cb_explore_adf -q UA --quiet --rnd 3 --epsilon 0.025")
	assert.NoError(t, err)
	acts := make([]*Action, 0)
	acts = append(acts, act1, act2)
	str := bnd.PredictLines(usr, acts)
	assert.EqualValues(t, []string{"|Action theme=sport", "|Action theme=music", "shared |User sex=f"}, str)
	acts, err = bnd.Predict(usr, acts)
	//predPrint(bnd, usr, acts)

	err = bnd.Reward(usr, acts, 0, .10)
	assert.NoError(t, err)
	acts, err = bnd.Predict(usr, acts)
	assert.NoError(t, err)
	assert.Equal(t, acts[0].Prob > acts[1].Prob, true)
	one := acts[0].Prob
	//predPrint(bnd, usr, acts)

	err = bnd.Reward(usr, acts, 0, .20)
	assert.NoError(t, err)
	acts, err = bnd.Predict(usr, acts)
	assert.NoError(t, err)
	assert.Equal(t, acts[0].Prob > one, true)
	twoId := acts[0].ID

	bnd.Reward(usr, acts, 0, .10)
	bnd.Reward(usr, acts, 0, .10)

	bnd.Reward(usr, acts, 1, .60)
	acts, err = bnd.Predict(usr, acts)
	assert.NoError(t, err)
	assert.Equal(t, true, acts[0].ID != twoId)

}

func predPrint(bnd *Bandit, c *Cntxt, act []*Action) {
	act, _ = bnd.Predict(c, act)
	for _, a := range act {
		fmt.Println(a.ID, a.Prob, a.ID)
	}
}

func Test_PMF(t *testing.T) {
	act := make([]*Action, 0)
	act = append(act, &Action{Prob: 0.55}, &Action{Prob: 0.45})
	id := Select(act)
	_ = id
	//fmt.Println(id, act[0], act[1])
}

// context usr tom ann
// time mon aft
// action pol mus spr
func getRew(c *Cntxt, act *Action) float32 {
	like := float32(1.0)
	dislike := float32(0.0)
	switch c.Prop["usr"] {
	case "tom":
		switch c.Prop["time"] {
		case "mon":
			if act.Prop["id"] == "politics" {
				return like
			}
			return dislike
		case "aft":
			if act.Prop["id"] == "music" {
				return like
			}
			return dislike
		default:
			return dislike
		}
	case "ann":
		switch c.Prop["time"] {
		case "mon":
			if act.Prop["id"] == "sports" {
				return like
			}
			return dislike
		case "aft":
			if act.Prop["id"] == "politics" {
				return like
			}
			return dislike
		default:
			return dislike
		}
	default:
		return dislike
	}
}

func cntxtRnd() *Cntxt {
	c := &Cntxt{Prop: make(map[string]string)}
	usr := []string{"ann", "tom"}
	time := []string{"mon", "aft"}
	c.Prop["usr"] = usr[rand.Intn(len(usr))]
	c.Prop["time"] = time[rand.Intn(len(time))]
	return c
}

func actRnd() (act []*Action) {
	ids := []string{"politics", "sports", "music", "food", "finance", "health", "camping"}
	for _, id := range ids {
		a := &Action{Prop: make(map[string]string)}
		a.Prop["id"] = id
		act = append(act, a)
	}

	return act
}

func Test_Tom(t *testing.T) {
	bnd, err := newBandit("--cb_explore_adf -q UA --epsilon 0.2 --no_stdin")
	//bnd, err := newBandit("--cb_explore_adf --q UA --cb_type dm --softmax --lambda 15")
	assert.NoError(t, err)
	defer bnd.vw.Finish()

	rew := float32(0)
	act := actRnd()
	for i := 0; i < 10001; i++ {
		c := cntxtRnd()

		act2, err := bnd.Predict(c, act)
		_ = act2
		if err != nil {
			log.Fatal(err)
		}
		//predPrint(bnd, c, act)
		sel := Select(act) //rand.Intn(len(act)) //Select(act)
		r := getRew(c, act[sel])
		//if r > 0 {
		//	fmt.Println(i, r, c.Prop["usr"], act[sel].Prop["id"])
		//}
		rew += r

		err = bnd.Reward(c, act, sel, r)
		if err != nil {
			log.Fatal(err)
		}

		if ((i < 500 && i%25 == 0) || i%500 == 0) && i > 0 {
			fmt.Println("rpm", i, rew/float32(i))
			//rew = 0
		}
	}
}

func Test_Explore(t *testing.T) {
	bnd, err := newBandit("--cb_explore 7 --cover 3  --no_stdin --quiet")
	//bnd, err := newBandit("--cb_explore_adf --q UA --cb_type dm --softmax --lambda 15")
	assert.NoError(t, err)
	defer bnd.vw.Finish()
	rew := float32(0)
	act := actRnd()
	for i := 0; i < 10000; i++ {
		c := cntxtRnd()
		//

		// | tom aft
		predStr := fmt.Sprintf(" | %s %s", c.Prop["usr"], c.Prop["time"])
		//fmt.Println(predStr)
		predictExample, err := bnd.vw.ReadExample(predStr)
		assert.NoError(t, err)
		pred := bnd.vw.Predict(predictExample)
		_ = pred

		scores := predictExample.GetActionScores()
		sel := weightedRandom(scores)

		r := getRew(c, act[sel])
		//fmt.Println("r", r)
		if r > 0 {
			//fmt.Println(i, r, c.Prop["usr"], act[sel].Prop["id"])
		}
		rew += r

		predictExample.Finish()

		example := fmt.Sprintf("%d:%f:%f | %s %s", (sel + 1), r*(-1), scores[sel], c.Prop["usr"], c.Prop["time"])
		//fmt.Println(example, sel, pred, scores)
		trainExample, err := bnd.vw.ReadExample(example)
		if err != nil {
			panic(err)
		}

		pr := bnd.vw.Learn(trainExample)
		_ = pr
		//fmt.Println("pr", pr)
		trainExample.Finish()
		if ((i < 300 && i%50 == 0) || i%500 == 0) && i > 0 {
			fmt.Println("rpm", i, rew/float32(i))
			//rew = 0
		}
	}

}

func weightedRandom(vals []float32) int {
	sum := float32(0)
	for _, val := range vals {
		sum += val
	}

	r := rand.Float32() * sum

	for i, val := range vals {
		r -= val
		if r <= 0 {
			return i
		}
	}

	return 0
}

func indexOf(val string, vals []string) int {
	for i, v := range vals {
		if val == v {
			return i
		}
	}

	return -1
}
