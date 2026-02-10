import React from 'react';
import { motion } from 'framer-motion';
import { Lock, Brain, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import PageTransition from '../components/PageTransition';

export default function QuizPage() {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-gray-50">
            <Navbar />
            <PageTransition>
                <div className="max-w-2xl mx-auto px-4 py-20 text-center">
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ type: 'spring', duration: 0.6 }}
                        className="mb-8"
                    >
                        <div className="relative inline-block">
                            <div className="w-24 h-24 bg-blue-50 rounded-3xl flex items-center justify-center mx-auto">
                                <Brain size={40} className="text-blue-300" />
                            </div>
                            <motion.div
                                className="absolute -bottom-1 -right-1 w-10 h-10 bg-white border-2 border-blue-200 rounded-xl flex items-center justify-center"
                                animate={{ rotate: [0, -10, 10, 0] }}
                                transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
                            >
                                <Lock size={16} className="text-blue-400" />
                            </motion.div>
                        </div>
                    </motion.div>

                    <motion.h1
                        className="text-2xl font-bold text-gray-900 mb-3"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        Quizzes Coming Soon
                    </motion.h1>

                    <motion.p
                        className="text-gray-500 max-w-md mx-auto mb-8 leading-relaxed"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3 }}
                    >
                        We're building adaptive math quizzes that adjust to your skill level.
                        Practice algebra, calculus, geometry and more with instant feedback.
                    </motion.p>

                    <motion.button
                        onClick={() => navigate('/dashboard')}
                        className="inline-flex items-center gap-2 px-6 py-2.5 text-sm font-medium text-blue-600 bg-blue-50 rounded-xl hover:bg-blue-100 transition-colors"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.4 }}
                    >
                        <ArrowLeft size={16} />
                        Back to Dashboard
                    </motion.button>
                </div>
            </PageTransition>
        </div>
    );
}
