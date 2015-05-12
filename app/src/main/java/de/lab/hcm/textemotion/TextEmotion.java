package de.lab.hcm.textemotion;

/**
 * Created by chi-tai on 05.05.15.

 * Chi-Tai Dang
 * HCM-Lab
 * University of Augsburg
 */

import android.annotation.SuppressLint;
import android.content.Context;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.AbstractInstance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

@SuppressLint("NewApi")
public class TextEmotion extends ActionBarActivity {

    static TextEmotion instance = null;
    private static TextView evalResult = null;
    private static ImageView resultText = null;
    private static EditText testText = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_emotion);

        instance = this;

        View v = findViewById(android.R.id.content);
        if (v == null) {
            android.util.Log.d(className, "onCreate: view not found!!!");
            return;
        }

        ViewGroup layout = (ViewGroup) v.findViewById(R.id.main_layout);
        if (layout == null) {
            return;
        }

        Button btnStart = (Button) layout.findViewById(R.id.train);
        if (btnStart == null) {
            return;
        }

        btnStart.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Train();
            }
        });

        btnStart = (Button) layout.findViewById(R.id.test);
        if (btnStart == null) {
            return;
        }

        btnStart.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                TestText();
            }
        });

        TextView tv = (TextView) layout.findViewById(R.id.evalResult);
        if (tv == null) {
            return;
        }
        evalResult = tv;

        ImageView iv = (ImageView) layout.findViewById(R.id.result);
        if (iv == null) {
            return;
        }
        resultText = iv;

        EditText et = (EditText) layout.findViewById(R.id.testText);
        if (et == null) {
            return;
        }
        testText = et;

        testText.setOnEditorActionListener(new TextView.OnEditorActionListener()
        {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event)
            {
                if (actionId == EditorInfo.IME_ACTION_DONE || (event != null && (event.getKeyCode() == KeyEvent.KEYCODE_ENTER))) {
                    InputMethodManager in = (InputMethodManager) instance.getSystemService(Context.INPUT_METHOD_SERVICE);

                    in.hideSoftInputFromWindow(v.getApplicationWindowToken(), 0);
                    v.clearFocus();
                    return true;

                }
                return false;
            }
        });

        resultText.setBackground(getResources().getDrawable(R.drawable.positive));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_text_emotion, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }



    static class Category {
        String name;
        int index;

        public String toString() {
            return index + " " + name;
        }
    }

    private final String className = "HCISheet5";


    private List<Category> categories = null;

    public Map<String, List<Integer>> catMap = null;

    // Category, Count
    public Map<Integer, Integer> resultMap = new HashMap<Integer, Integer>();


    private static int ATTRIB_COUNT = 2;
    private static int ATTRIB_CLASS = 0;
    public static String[] emotions = null;
    public static FastVector attributes = null;

    private Classifier cl;
    private Instances trainingData;

    private void Train ()
    {
        android.util.Log.d(className, "Training...");

        try {
            emotions = new String[] {"Positive", "Negative", "Neutral"};

            ATTRIB_COUNT = 0;

            FastVector fvClassVal = new FastVector(2);

            fvClassVal.addElement(emotions[0]);
            fvClassVal.addElement(emotions[1]);
            fvClassVal.addElement(emotions[2]);

            Attribute emotionClasses = new Attribute("Emotion", fvClassVal);
            ATTRIB_COUNT++;

            Attribute myFeature = new Attribute("myFeature");
            ATTRIB_COUNT++;

            // Create vector for attributes
            attributes = new FastVector(ATTRIB_COUNT);
            attributes.addElement(emotionClasses);
            ATTRIB_CLASS = 0;

            attributes.addElement(myFeature);

            // Load category map
            catMap = DataParser.loadCatMap();

            // Load categories
            categories = DataParser.loadCategories();

            // Load training set
            List<String> neutrals = DataParser.loadDataLines("Neutral.txt"),
                    positives = DataParser.loadDataLines("Positive.txt"),
                    negatives = DataParser.loadDataLines("Negative.txt");


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

            fw.write(trainingData.toString()); fw.flush(); fw.close();

            cl = (Classifier) new NaiveBayes();

            // Train out classifier
            cl.buildClassifier(trainingData);

            // Test the model
            Evaluation eva = new Evaluation(trainingData);

            // 10-fold cross validation
            eva.crossValidateModel(cl, trainingData, 10, new Random(10));

            if (evalResult != null) {
                String evals = eva.toSummaryString();
                evals += "\r\n" + eva.toClassDetailsString();
                evals += "\r\n" + eva.toMatrixString();
                evalResult.setText (evals);
            }

            System.out.println(eva.toSummaryString());
            System.out.println(eva.toClassDetailsString());
            System.out.println(eva.toMatrixString());
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }


    private void TestText()
    {
        String line = testText.getText().toString();
        System.out.println("Texting: " + line);

        double[] result = new double[0];
        try {
            result = classifyString(line);

            // Output emotion
            System.out.println(emotions[0] + ": " + result[0]);
            System.out.println(emotions[1] + ": " + result[1]);
            System.out.println(emotions[2] + ": " + result[2]);

            double max = 0; int sel = 0;
            for(int i=0; i<3; i++) if(result[i]>max) {max=result[i]; sel = i; }
            System.out.println("Ergebnis: " + emotions[sel]);

            if (result != null) {
                switch (sel) {
                    case 0:
                        resultText.setBackground(getResources().getDrawable(R.drawable.positive));
                        break;
                    case 1:
                        resultText.setBackground(getResources().getDrawable(R.drawable.negative));
                        break;
                    case 2:
                        resultText.setBackground(getResources().getDrawable(R.drawable.neutral));
                        break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    // Used to fill our dataset with instances and according features
    public void classifyMore(Instances ret, List<String> strings, String emotion) {
        for(String string: strings)
            ret.add(getWEKAInstance(string, emotion));
    }

    // Used to classify a new text using our trained classifier
    private double[] classifyString(String text) throws Exception {
        Instance iUse = getWEKAInstance(text, null);
        iUse.setDataset(trainingData);
        return cl.distributionForInstance(iUse);
    }

    //
    // This is the "hot spot" where feature extraction has to be done...
    //
    public Instance getWEKAInstance(String text, String emotion) {
        //
        // Assign the first attributes (the class)
        Instance ret = new DenseInstance(ATTRIB_COUNT);
        if(emotion != null)
            ret.setValue((Attribute)attributes.elementAt(0), emotion);

        //
        // Do some feature extraction and calculation
        // Make use of the dictionary and word categories
        //
        float exampleValue = 0.0f;

        // e.g. we use the length of the string
        exampleValue = text.length();

        ret.setValue((Attribute)attributes.elementAt(1), exampleValue);

        return ret;
    }
}
