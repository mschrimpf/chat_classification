package de.lab.hcm.textemotion;

/**
 * Created by chi-tai on 05.05.15.
 * <p/>
 * Chi-Tai Dang
 * HCM-Lab
 * University of Augsburg
 */

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Pattern;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

@SuppressLint("NewApi")
public class TextEmotionActivity extends Activity {

    private static final String TAG = "TextEmotionActivity";

    private TextView mEvalResultTextView = null;
    private ImageView mResultImageView = null;
    private EditText mTestInputEditText = null;
    private DataParser mDataParser;

    private List<Category> categories = null;
    public SortedMap<String, List<Integer>> catMap = null;

    // Category, Count
    public Map<Integer, Integer> resultMap = new HashMap<>();
    private int ATTRIB_COUNT = 2;
    private int ATTRIB_CLASS = 0;
    public String[] emotions = null;
    public FastVector<Attribute> attributes = null;
    private Classifier cl;
    private Instances trainingData;

    private List<Integer> negativeList;
    private List<Integer> positiveList;
    private List<String> positiveSmileys;
    private List<String> negativeSmileys;


    //Negation value of LIWCCategories
    int negation = 7;
    int exclusive = 45;
    int tentative = 25; // maybe, perhaps, gues
    int certainty = 26;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_emotion);
        mDataParser = new DataParser(this);

        mEvalResultTextView = (TextView) findViewById(R.id.evalResult);
        mResultImageView = (ImageView) findViewById(R.id.result);
        mTestInputEditText = (EditText) findViewById(R.id.testText);
        mTestInputEditText.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                if (actionId == EditorInfo.IME_ACTION_DONE || (event != null && (event.getKeyCode() == KeyEvent.KEYCODE_ENTER))) {
                    InputMethodManager in = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);

                    in.hideSoftInputFromWindow(v.getApplicationWindowToken(), 0);
                    v.clearFocus();
                    return true;

                }
                return false;
            }
        });
        mTestInputEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                testText();
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        mResultImageView.setBackground(getResources().getDrawable(R.drawable.neutral, null));
        initLists();
        train();

        // comment this in for finding best solutions
        //evaluatePossibilities();
        //count(notRecognizedWords);
    }

    private void evaluatePossibilities() {
        negativeList = new ArrayList<>();
        positiveList = new ArrayList<>();
        Collections.addAll(negativeList, 16,18,19,59,66
                );
        Collections.addAll(positiveList, 3,13,14,15
                );
        double score = train();

        for (Integer i = 1; i < 68; i++) {
            if (negativeList.contains(i) || positiveList.contains(i)){
                continue;
            }
            negativeList.add(i);
            double scoreNegativeAdd = train();
            negativeList.remove(i);
            positiveList.add(i);
            double scorePositiveAdd = train();
            negativeList.add(i);
            double scoreBothAdd = train();
            negativeList.remove(i);
            positiveList.remove(i);

            if (score > scoreNegativeAdd || score > scorePositiveAdd || score > scoreBothAdd) {
                if (scoreNegativeAdd < scorePositiveAdd) {
                    if (scoreNegativeAdd < scoreBothAdd) {
                        negativeList.add(i);
                        score = scoreNegativeAdd;
                    } else {
                        positiveList.add(i);
                        negativeList.add(i);
                        score = scoreBothAdd;
                    }
                } else if (scorePositiveAdd < scoreBothAdd) {
                    positiveList.add(i);
                    score = scorePositiveAdd;
                } else {
                    positiveList.add(i);
                    negativeList.add(i);
                    score = scoreBothAdd;
                }
            }
            Log.w(TAG, "Success rate for run " + i + " is: " + (1 - score));
        }
        Log.w(TAG, "Error rate is: " + score);
    }

    private void initLists() {
        //Negative values of LIWCCategories
        negativeList = new ArrayList<>();
        Collections.addAll(negativeList
                , 16 //negemo - negative emotions
                , 17 //anx - anxiety
                , 18 //anger - anger
                , 19 //sad - sadness
                //,24 //inhib - Inhibition
                //,40 //future
                //,61 //body
                //,43 //down
                //,47 //occup
                //,48 //school
                //,49 //job
                //,56 //money
                //,58 //religion
                , 59 //death
                , 64 //
                , 66 //swear - Swear Words
                //, 67 //Nonfl - Nonfluencies
                //,68  //Fillers
        );

        //Positive values of LIWCCategories
        positiveList = new ArrayList<>();
        Collections.addAll(positiveList
                , 3 //we
                //,8 //assent - Agree, OK, yes
                , 13 //posemo - positive emotions
                , 14    //Posfeel - positive feelings
                , 15    //Optim - optimism
                //,31 //social
                //,32 //comm - communication
                //, 34 //Friends
                , 35 //Family
                , 36 //humans
                //,38 //past
                //,42 //up
                , 50 //achieve
                //,51 //leisure
                //,52 //home
                //,53 //sports
                //,54 //tv
                //,55 //Music
                //,62 //sexual
                //,63 //eating
                //,64 //sleep
                //,65 //groom
        );
    }

    List<String> notRecognizedWords;

    private double train() {
        notRecognizedWords = new ArrayList<>();
        Log.d(TAG, "Training...");

        try {
            emotions = new String[]{"Positive", "Negative", "Neutral"};

            ATTRIB_COUNT = 0;

            FastVector<String> fvClassVal = new FastVector<>(2);
            fvClassVal.addElement(emotions[0]);
            fvClassVal.addElement(emotions[1]);
            fvClassVal.addElement(emotions[2]);

            Attribute emotionClasses = new Attribute("Emotion", fvClassVal);
            ATTRIB_COUNT++;
            Attribute liwcCategories = new Attribute("liwcCategories");
            ATTRIB_COUNT++;
            Attribute wordAnalysis = new Attribute("wordAnalysis");
            ATTRIB_COUNT++;

            // Create vector for attributes
            attributes = new FastVector<>(ATTRIB_COUNT);
            attributes.addElement(emotionClasses);
            attributes.addElement(liwcCategories);
            attributes.addElement(wordAnalysis);
            ATTRIB_CLASS = 0;

            // Load category map
            catMap = mDataParser.loadCatMap();

            // Load categories
            categories = mDataParser.loadCategories();

            // Load training set
            List<String> neutrals = mDataParser.loadDataLines("Neutral.txt"),
                    positives = mDataParser.loadDataLines("Positive.txt"),
                    negatives = mDataParser.loadDataLines("Negative.txt");


            // Create weka instances and name them Text2Emo
            trainingData = new Instances("Text2Emo", attributes, positives.size());

            /// Our first attribute (0) declares the class
            trainingData.setClassIndex(ATTRIB_CLASS);

            classifyMore(trainingData, positives, "Positive");
            classifyMore(trainingData, neutrals, "Neutral");
            classifyMore(trainingData, negatives, "Negative");

            // Backup weka instances
            File trainFile = File.createTempFile("Training", ".arff");
            FileWriter fw = new FileWriter(trainFile);

            fw.write(trainingData.toString());
            fw.flush();
            fw.close();

            cl = new NaiveBayes();

            // train out classifier
            cl.buildClassifier(trainingData);

            // Test the model
            Evaluation eva = new Evaluation(trainingData);

            // 10-fold cross validation
            eva.crossValidateModel(cl, trainingData, 10, new Random(10));

            if (mEvalResultTextView != null) {
                String evals = eva.toSummaryString();
                evals += "\r\n" + eva.toClassDetailsString();
                evals += "\r\n" + eva.toMatrixString();
                mEvalResultTextView.setText(evals);
            }
            Log.d(TAG, eva.toSummaryString());
            Log.d(TAG, eva.toClassDetailsString());
            Log.d(TAG, eva.toMatrixString());
            return eva.errorRate();
        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }


    private void testText() {
        String line = mTestInputEditText.getText().toString();
        Log.d(TAG, "Texting: " + line);

        try {
            double[] result = classifyString(line);

            // Output emotion
            Log.d(TAG, emotions[0] + ": " + result[0]);
            Log.d(TAG, emotions[1] + ": " + result[1]);
            Log.d(TAG, emotions[2] + ": " + result[2]);

            double max = 0;
            int sel = 0;
            for (int i = 0; i < 3; i++) {
                if (result[i] > max) {
                    max = result[i];
                    sel = i;
                }
            }
            Log.d(TAG, "Ergebnis: " + emotions[sel]);
            final int value = sel;
            mResultImageView.post(new Runnable() {
                @Override
                public void run() {

                    switch (value) {
                        case 0:
                            mResultImageView.setBackground(getResources().getDrawable(R.drawable.positive, null));
                            break;
                        case 1:
                            mResultImageView.setBackground(getResources().getDrawable(R.drawable.negative, null));
                            break;
                        case 2:
                            mResultImageView.setBackground(getResources().getDrawable(R.drawable.neutral, null));
                            break;
                    }
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    // Used to fill our dataset with instances and according features
    public void classifyMore(Instances ret, List<String> strings, String emotion) {
        for (String string : strings) {
            ret.add(getWEKAInstance(string, emotion));
        }
    }

    // Used to classify a new text using our trained classifier
    private double[] classifyString(String text) throws Exception {
        Instance iUse = getWEKAInstance(text, null);
        iUse.setDataset(trainingData);
        return cl.distributionForInstance(iUse);
    }

    /*
     * This is the "hot spot" where feature extraction has to be done...
     */
    public Instance getWEKAInstance(String text, String emotion) {
        text = text.trim();
        // Assign the first attributes (the class)
        Instance ret = new DenseInstance(ATTRIB_COUNT);
        if (emotion != null) {
            ret.setValue(attributes.elementAt(0), emotion);
        }

        //
        // Do some feature extraction and calculation
        // Make use of the dictionary and word categories
        //

        /* Gets all matches */

        List<Integer> matchingCategories = new ArrayList<>();

        String[] wordSplit = text.toLowerCase().split(" ");
        for (String word : wordSplit) {
            if (word == null || Objects.equals(word, "")) {
                continue;
            }
            if (catMap.containsKey(word)) {
                /* already inside of the map */
                matchingCategories.addAll(catMap.get(word));
            } else {
                /* Look for every word which starts with first two (or one, if only one character) and check those */
                String prefix = word.length() > 1 ? word.substring(0, 2) : word.substring(0, 1);
                boolean matched = false;
                for (Map.Entry<String, List<Integer>> entry : filterPrefix(catMap, prefix).entrySet()) {
                    Pattern pattern = Pattern.compile(entry.getKey());
                    if (pattern.matcher(word).matches()) {
                        matchingCategories.addAll(entry.getValue());
                        matched = true;
                    }
                }
                if (!matched) {
                    String[] escapedWords = word.split("\\W");
                    for (String escapedWord : escapedWords) {
                        if (escapedWord == null || Objects.equals(escapedWord, "")) {
                            continue;
                        }
                        String escapedPrefix = escapedWord.length() > 1 ? escapedWord.substring(0, 2) : escapedWord.substring(0, 1);
                        for (Map.Entry<String, List<Integer>> entry : filterPrefix(catMap, escapedPrefix).entrySet()) {
                            Pattern pattern = Pattern.compile(entry.getKey());
                            if (pattern.matcher(escapedWord).matches()) {
                                matchingCategories.addAll(entry.getValue());
                                matched = true;
                            }
                        }
                        if (!matched) {
                            notRecognizedWords.add(escapedWord);
                        }
                    }
                }
            }
        }


        //matchingCategories contains all categories that are found in the given text - now what?! TODO
        int value = 0;
        boolean negated = false;
        int effect = 1;
        for (int categoryId : matchingCategories) {
            if (positiveList.indexOf(categoryId) >= 0) {
                value += 1;
            }
            if (negativeList.indexOf(categoryId) >= 0) {
                value -= 1;
            }
            if (categoryId == negation){
                negated = !negated;
            }
            /*
            //maybe, perhaps...
            if (categoryId == tentative){
                effect *= 0.5;
            } if (categoryId == 2 || categoryId == 4 || categoryId == 3){ // I, myself, we
                //effect *= 0.5;
            }
            if (categoryId == 5 || categoryId == 6){ //you, others
                effect *= 0.9;
            }*/
        }
        if(text.matches(".*\\?\\W*$") && negated) {
            negated = false;
        }
        if (negated){
            value *= -1;
        }
        value *= effect;

        ret.setValue(attributes.elementAt(1), value); //liwcCategories
        //ret.setValue(attributes.elementAt(2), exampleValue); //wordAnalysis
        return ret;
    }

    private void count(Collection<String> collection) {
        final Map<String, Integer> map = new TreeMap<>();
        for (String string : collection) {
            if (map.containsKey(string)) {
                map.put(string, map.get(string) + 1);
            } else {
                map.put(string, 1);
            }
        }
        SortedSet<Map.Entry<String, Integer>> sortedMap = entriesSortedByValues(map);
        for (Map.Entry<String, Integer> pair : sortedMap) {
            Log.i(TAG, pair.getKey() + ": " + pair.getValue());
        }
    }

    static <K, V extends Comparable<? super V>>
    SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map) {
        SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<>(
                new Comparator<Map.Entry<K, V>>() {
                    @Override
                    public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
                        int res = e1.getValue().compareTo(e2.getValue());
                        return res != 0 ? res : 1;
                    }
                }
        );
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    /* http://stackoverflow.com/questions/6713239/partial-search-in-hashmap */
    public <V> SortedMap<String, V> filterPrefix(SortedMap<String, V> baseMap, String prefix) {
        if (prefix.length() > 0) {
            char nextLetter = prefix.charAt(prefix.length() - 1);
            nextLetter++;
            String end = prefix.substring(0, prefix.length() - 1) + nextLetter;
            return baseMap.subMap(prefix, end);
        }
        return baseMap;
    }
}
